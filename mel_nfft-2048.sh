#!/opt/conda/bin/python
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -m 128 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -m 256 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -m 512 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -m 1024 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -m 2048 -g 3

