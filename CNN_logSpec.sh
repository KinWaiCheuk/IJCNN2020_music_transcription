#!/opt/conda/bin/python
python CNN_logSpec.py --n_fft 4096 -g 2 -e 11
python CNN_logSpec.py --n_fft 2048 -g 2 -e 11
python CNN_logSpec.py --n_fft 1024 -g 2 -e 11
python CNN_logSpec.py --n_fft 512 -g 2 -e 11
python CNN_logSpec.py --n_fft 256 -g 2 -e 11
