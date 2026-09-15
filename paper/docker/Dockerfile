# TCSVT 원고 컴파일 환경.
#   docker build -t sctex:ieee paper/docker
# 베이스 이미지에 없는 패키지만 얹는다. IEEEtran.cls/.bst는 이미지가 아니라
# paper/vendor/IEEEtran/ 에 저장소와 함께 고정돼 있다(재현성).
FROM texlive/texlive:latest-medium

# multirow  : 표의 세로 병합
# orcidlink : 저자 ORCID 아이콘 (IEEE가 요구)
# preprint  : balance.sty (마지막 쪽 두 단 높이 맞춤)
RUN tlmgr --repository https://mirror.ctan.org/systems/texlive/tlnet install \
        multirow orcidlink preprint \
 && mktexlsr

WORKDIR /work
