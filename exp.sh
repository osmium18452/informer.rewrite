gpu=2
school="--fudan"

set -x

python main.py -b 320 $school -Gc $gpu -e 10 -S 1 -s save/23.5.31/01 -p 24
python main.py -b 200 $school -Gc $gpu -e 10 -S 1 -s save/23.5.31/02 -p 48
python main.py -b 120 $school -Gc $gpu -e 10 -S 1 -s save/23.5.31/03 -p 96
python main.py -b 60  $school -Gc $gpu -e 10 -S 1 -s save/23.5.31/04 -p 168
python main.py -b 50  $school -Gc $gpu -e 10 -S 1 -s save/23.5.31/05 -p 200
python main.py -b 32  $school -Gc $gpu -e 10 -S 1 -s save/23.5.31/06 -p 336
python main.py -b 320 $school -Gc $gpu -e 10 -S 5 -s save/23.5.31/11 -p 24
python main.py -b 200 $school -Gc $gpu -e 10 -S 5 -s save/23.5.31/12 -p 48
python main.py -b 120 $school -Gc $gpu -e 10 -S 5 -s save/23.5.31/13 -p 96
python main.py -b 60  $school -Gc $gpu -e 10 -S 5 -s save/23.5.31/14 -p 168
python main.py -b 50  $school -Gc $gpu -e 10 -S 5 -s save/23.5.31/15 -p 200
python main.py -b 32  $school -Gc $gpu -e 10 -S 5 -s save/23.5.31/16 -p 336
python main.py -b 320 $school -Gc $gpu -e 10 -S 10 -s save/23.5.31/21 -p 24
python main.py -b 200 $school -Gc $gpu -e 10 -S 10 -s save/23.5.31/22 -p 48
python main.py -b 120 $school -Gc $gpu -e 10 -S 10 -s save/23.5.31/23 -p 96
python main.py -b 60  $school -Gc $gpu -e 10 -S 10 -s save/23.5.31/24 -p 168
python main.py -b 50  $school -Gc $gpu -e 10 -S 10 -s save/23.5.31/25 -p 200
python main.py -b 32  $school -Gc $gpu -e 10 -S 10 -s save/23.5.31/26 -p 336
python main.py -b 320 $school -Gc $gpu -e 10 -S 20 -s save/23.5.31/31 -p 24
python main.py -b 200 $school -Gc $gpu -e 10 -S 20 -s save/23.5.31/32 -p 48
python main.py -b 120 $school -Gc $gpu -e 10 -S 20 -s save/23.5.31/33 -p 96
python main.py -b 60  $school -Gc $gpu -e 10 -S 20 -s save/23.5.31/34 -p 168
python main.py -b 50  $school -Gc $gpu -e 10 -S 20 -s save/23.5.31/35 -p 200
python main.py -b 32  $school -Gc $gpu -e 10 -S 20 -s save/23.5.31/36 -p 336
# epochs=100
python main.py -b 320 $school -Gc $gpu -e 100 -S 1 -s save/23.5.31/101 -p 24
python main.py -b 200 $school -Gc $gpu -e 100 -S 1 -s save/23.5.31/102 -p 48
python main.py -b 120 $school -Gc $gpu -e 100 -S 1 -s save/23.5.31/103 -p 96
python main.py -b 60  $school -Gc $gpu -e 100 -S 1 -s save/23.5.31/104 -p 168
python main.py -b 50  $school -Gc $gpu -e 100 -S 1 -s save/23.5.31/105 -p 200
python main.py -b 32  $school -Gc $gpu -e 100 -S 1 -s save/23.5.31/106 -p 336
python main.py -b 320 $school -Gc $gpu -e 100 -S 5 -s save/23.5.31/111 -p 24
python main.py -b 200 $school -Gc $gpu -e 100 -S 5 -s save/23.5.31/112 -p 48
python main.py -b 120 $school -Gc $gpu -e 100 -S 5 -s save/23.5.31/113 -p 96
python main.py -b 60  $school -Gc $gpu -e 100 -S 5 -s save/23.5.31/114 -p 168
python main.py -b 50  $school -Gc $gpu -e 100 -S 5 -s save/23.5.31/115 -p 200
python main.py -b 32  $school -Gc $gpu -e 100 -S 5 -s save/23.5.31/116 -p 336
python main.py -b 320 $school -Gc $gpu -e 100 -S 10 -s save/23.5.31/121 -p 24
python main.py -b 200 $school -Gc $gpu -e 100 -S 10 -s save/23.5.31/122 -p 48
python main.py -b 120 $school -Gc $gpu -e 100 -S 10 -s save/23.5.31/123 -p 96
python main.py -b 60  $school -Gc $gpu -e 100 -S 10 -s save/23.5.31/124 -p 168
python main.py -b 50  $school -Gc $gpu -e 100 -S 10 -s save/23.5.31/125 -p 200
python main.py -b 32  $school -Gc $gpu -e 100 -S 10 -s save/23.5.31/126 -p 336
python main.py -b 320 $school -Gc $gpu -e 100 -S 20 -s save/23.5.31/131 -p 24
python main.py -b 200 $school -Gc $gpu -e 100 -S 20 -s save/23.5.31/132 -p 48
python main.py -b 120 $school -Gc $gpu -e 100 -S 20 -s save/23.5.31/133 -p 96
python main.py -b 60  $school -Gc $gpu -e 100 -S 20 -s save/23.5.31/134 -p 168
python main.py -b 50  $school -Gc $gpu -e 100 -S 20 -s save/23.5.31/135 -p 200
python main.py -b 32  $school -Gc $gpu -e 100 -S 20 -s save/23.5.31/136 -p 336
