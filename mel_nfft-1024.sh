#!/opt/conda/bin/python
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -w 128 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -w 256 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -w 512 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -w 1024 -g 3
python LinearLayer_Melspectrogram.py --n_fft 2048 -g 1 -w 2048 -g 3

