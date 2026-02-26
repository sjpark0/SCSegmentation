import PyNvVideoCodec as nvc
import numpy as np
import torch
# import pycuda.driver as cuda
# import pycuda.autoinit
import logging
import os
import subprocess
import json
import cv2
import time
import threading

class VideoDecoder:
    def __init__(self, codec=nvc.cudaVideoCodec.H264, gpuid=0, usedevicememory=True):
        self.codec = codec
        self.gpuid = gpuid
        self.usedevicememory = usedevicememory
        self.decoder = None
        self.demuxer = None
        self.frame_count = 0
        self.packet_iterator = None
        self.frame_iterator = None

    def initialize(self, input_file):
        self.frame_count = 0
        self.demuxer = nvc.CreateDemuxer(filename=input_file)
        self.decoder = nvc.CreateDecoder(
            gpuid=self.gpuid,
            codec=self.codec,
            cudacontext=0,
            cudastream=0,
            usedevicememory=self.usedevicememory
        )
        self.packet_iterator = iter(self.demuxer)
        self.frame_iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame_iterator is None:
            try:
                packet = next(self.packet_iterator)
                self.frame_iterator = iter(self.decoder.Decode(packet))
            except StopIteration:
                raise StopIteration
            except Exception as e:
                logging.error(f'Error decoding packet: {e}', exc_info=True)
                raise e
        
        try:
            decoded_frame = next(self.frame_iterator)
            self.frame_count += 1
            return self.process_frame(decoded_frame)
        except StopIteration:
            self.frame_iterator = None
            return self.__next__()
        except Exception as e:
            logging.error(f'Error decoding frame: {e}', exc_info=True)
            raise e

    @staticmethod
    def nv12_to_rgb(nv12_tensor, width, height):
        try:
            nv12_tensor = nv12_tensor.to(dtype=torch.float32)
            y_plane = nv12_tensor[:height, :width]
            uv_plane = nv12_tensor[height:height + height // 2, :].view(height // 2, width // 2, 2).repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
            u_plane = uv_plane[:, :, 0] - 128
            v_plane = uv_plane[:, :, 1] - 128
            r = y_plane + 1.402 * v_plane
            g = y_plane - 0.344136 * u_plane - 0.714136 * v_plane
            b = y_plane + 1.772 * u_plane
            rgb_frame = torch.stack((b, g, r), dim=2).clamp(0, 255).byte()
            return rgb_frame
        except Exception as e:
            logging.error(f'Error converting NV12 to RGB: {e}', exc_info=True)
            raise e

    def process_frame(self, frame):
        try:
            src_tensor = torch.from_dlpack(frame)
            (height, width) = frame.shape
            rgb_tensor = self.nv12_to_rgb(src_tensor, width, int(height / 1.5))
            return rgb_tensor
        except Exception as e:
            logging.error(f'Error processing frame: {e}', exc_info=True)
            raise e
        
class VideoEncoder:
    def __init__(self, width, height, format, use_cpu_input_buffer=False, **kwargs):
        self.width = width
        self.height = height
        self.format = format
        self.use_cpu_input_buffer = use_cpu_input_buffer
        self.encoder = nvc.CreateEncoder(width, height, format, use_cpu_input_buffer, **kwargs)
        logging.info(f'Encoder created with width: {width}, height: {height}, format: {format}, use_cpu_input_buffer: {use_cpu_input_buffer}')

    @staticmethod
    def rgb_to_yuv(rgb_tensor):
        rgb_tensor = rgb_tensor.to(dtype=torch.float32)
        r = rgb_tensor[:, :, 0]
        g = rgb_tensor[:, :, 1]
        b = rgb_tensor[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128
        height, width = rgb_tensor.shape[:2]
        y_plane = y
        u_plane = u[0::2, 0::2]
        v_plane = v[0::2, 0::2]
        uv_plane = torch.stack((u_plane, v_plane), dim=2).reshape(height // 2, width)
        tensor_yuv = torch.cat((y_plane, uv_plane), dim=0).clamp(0, 255).byte()
        return tensor_yuv

    def encode(self, input_data):
        try:
            bitstream = self.encoder.Encode(input_data)
            return bitstream
        except Exception as e:
            logging.error(f'Error encoding frame: {e}', exc_info=True)
            return None

    def end_encode(self):
        try:
            bitstream = self.encoder.EndEncode()
            logging.info('Encoder flushed successfully')
            return bitstream
        except Exception as e:
            logging.error(f'Error ending encode: {e}', exc_info=True)
            return None

    def reconfigure(self, params):
        try:
            self.encoder.Reconfigure(params)
            logging.info('Encoder reconfigured successfully')
        except Exception as e:
            logging.error(f'Error reconfiguring encoder: {e}', exc_info=True)

    def get_reconfigure_params(self):
        try:
            params = self.encoder.GetEncodeReconfigureParams()
            logging.info('Reconfigure parameters fetched successfully')
            return params
        except Exception as e:
            logging.error(f'Error fetching reconfigure parameters: {e}', exc_info=True)
            return None


def process(input_folder, video_encoder, output_file):
    fp = open(output_file, 'wb')
    for i in range(50):
        filename = input_folder + "/{:d}.jpg".format(i)
        img = cv2.imread(filename)
        input_tensor = video_encoder.rgb_to_yuv(torch.tensor(img))
        input_tensor = input_tensor.cpu()
        bitstream = video_encoder.encode(input_tensor)
        if bitstream:
            fp.write(bytearray(bitstream))
    remaining_bitstream = video_encoder.end_encode()
    if remaining_bitstream:
        fp.write(bytearray(remaining_bitstream))
    fp.close()

def get_video_info(video_path):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)
    width = info['streams'][0]['width']
    height = info['streams'][0]['height']
    #duration = float(info['streams'][0]['duration'])
    return width, height

def encode_test():
    input_folder = "../../Data/VideoSample/0"
    output_file = "../../Data/VideoSample_HEVC/0.hevc"
    
    img = cv2.imread(input_folder + "/0.jpg")
    height, width, _ = img.shape
    duration = 10
    
    video_encoder = VideoEncoder(width=width, height=height, format="NV12", use_cpu_input_buffer=False, codec="hevc", bitrate=4000000, fps=30)
    
    process(input_folder, video_encoder, output_file)

def decoding_thread(video_decoder, dataready, rgb_tensors, finished, decoder_idx):
    while not finished[decoder_idx]:
        if not dataready[decoder_idx]:
            rgb_tensors[decoder_idx] = next(video_decoder)
            dataready[decoder_idx] = True
            if rgb_tensors[decoder_idx] is None:
                break
        else:
            time.sleep(0.001)

def decode_test():
    input_files = [None] * 11
    video_decoders = [None] * 11
    dataready = [False] * 11
    finished = [False] * 11
    threads = [None] * 11
    rgb_tensors = [None] * 11

    for i in range(11):
        input_files[i] = "../../Data/VideoSample_HEVC/Image_0.HEVC"
        video_decoders[i] = VideoDecoder(codec=nvc.cudaVideoCodec.HEVC)
        video_decoders[i].initialize(input_files[i])
        threads[i] = threading.Thread(target=decoding_thread, args=(video_decoders[i], dataready, rgb_tensors, finished, i))

    for i in range(11):
        threads[i].start()
    
    frame_idx = 0
    start = time.time()
    width = 7680
    height = 4320
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    
    all_dataready = False    
    while frame_idx < 1000:
        while not all_dataready:
            all_dataready = True
            for i in range(11):
                all_dataready = all_dataready and dataready[i]
            time.sleep(0.001)

        img = torch.stack(rgb_tensors, dim=0)
        avg_img = torch.mean(img, dim=0, dtype=torch.float16).cpu().numpy()
        
        #cv2.imshow('image', avg_img)    
        #cv2.resizeWindow('image', 960, 540)
        #cv2.waitKey(1)
        
        elapsed = time.time() - start
        print(f'Frame {frame_idx} decoded in {elapsed:.2f} seconds')
        frame_idx += 1
        for i in range(11):
            dataready[i] = False
        #time.sleep(0.001)
    

    for i in range(11):
        finished[i] = True
        threads[i].join()

    cv2.destroyAllWindows()
    #output_folder = "../../Data/VideoSample_HEVC/Image_0"
    #os.makedirs(output_folder, exist_ok=True)
    
    #width = 7680
    #height = 4320
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #start = time.time()
    #for i in range(1000):
    #    for j in range(11):
    #        rgb_tensors[j] = next(video_decoders[j])
        
    #    elapsed = time.time() - start
    #    print(f'Frame {i} decoded in {elapsed:.2f} seconds')
        
    #for frame_idx, rgb_tensor in enumerate(video_decoder):
    #    elapsed = time.time() - start
    #    print(f'Frame {frame_idx} decoded in {elapsed:.2f} seconds')
        #img = rgb_tensor.cpu().numpy()
        #filename = output_folder + "/{:d}.jpg".format(frame_idx)
        #cv2.imwrite(filename, img)
        #cv2.imshow('image', img)    
        #cv2.resizeWindow('image', 960, 540)
    #    cv2.waitKey(1)
    #cv2.destroyAllWindows()
#encode_test()
decode_test()
