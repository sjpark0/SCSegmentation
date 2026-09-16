# TCSVT 원고 빌드. 호스트에 TeX을 깔지 않고 도커로만 돌린다.
#
#   make            원고 PDF 생성 (main.pdf)
#   make pages      쪽수 출력
#   make final      \draftfalse 로 빌드해 final.pdf 생성 (TODO 주석 숨김)
#   make clean      중간 파일 삭제
#   make image      도커 이미지 빌드 (최초 1회)
#   make bundle     제출 묶음: submission/tcsvt-manuscript.pdf (TODO 숨김) +
#                   submission/tcsvt-source.zip (tex·bib·bbl·cls·bst·figures, 평평한 구조)

IMAGE  := sctex:ieee
UID    := $(shell id -u)
GID    := $(shell id -g)
RUN     = docker run --rm --user $(UID):$(GID) \
            -v $(CURDIR):/work -w /work \
            -e HOME=/tmp \
            -e TEXINPUTS=".//:./vendor/IEEEtran//:" \
            -e BSTINPUTS=".//:./vendor/IEEEtran//:" \
            -e BIBINPUTS=".//:" \
            $(IMAGE)

.PHONY: all pages final clean image bundle

all: main.pdf

main.pdf: main.tex numbers.tex refs.bib
	$(RUN) latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# 쪽수는 pdflatex 로그에서 읽는다 (이미지에 pdfinfo 가 없다)
pages: main.pdf
	@sed -n 's/.*Output written on main\.pdf (\([0-9]*\) pages.*/main.pdf: \1 pages/p' main.log | tail -1

final: main.tex numbers.tex refs.bib
	@sed 's/\\drafttrue/\\draftfalse/' main.tex > final.tex
	$(RUN) latexmk -pdf -interaction=nonstopmode -halt-on-error final.tex
	@sed -n 's/.*Output written on final\.pdf (\([0-9]*\) pages.*/final.pdf: \1 pages/p' final.log | tail -1

clean:
	$(RUN) latexmk -C main.tex final.tex || true
	rm -f final.tex *.bbl *.blg *.aux *.log *.out *.fls *.fdb_latexmk

image:
	docker build -t $(IMAGE) docker

# 제출 묶음. IEEE 저널은 초기 투고에 PDF 하나, 채택 후에 소스(.tex/.bib/.bbl/그림)를 요구한다.
# .bbl 을 함께 넣는 이유: IEEE 조판 시스템은 bibtex 을 다시 돌리지 않는 경우가 있다.
bundle: final
	rm -rf submission && mkdir -p submission/src/figures
	cp final.pdf submission/tcsvt-manuscript.pdf
	cp final.tex submission/src/main.tex
	cp numbers.tex refs.bib submission/src/
	cp final.bbl submission/src/main.bbl
	cp vendor/IEEEtran/IEEEtran.cls vendor/IEEEtran/IEEEtran.bst submission/src/
	cp figures/fig_*.pdf submission/src/figures/
	cd submission/src && zip -q -r ../tcsvt-source.zip .
	@echo "submission/tcsvt-manuscript.pdf  $$(du -h submission/tcsvt-manuscript.pdf | cut -f1)"
	@echo "submission/tcsvt-source.zip      $$(du -h submission/tcsvt-source.zip | cut -f1)"
	@unzip -l submission/tcsvt-source.zip | tail -n +4 | head -n -2
