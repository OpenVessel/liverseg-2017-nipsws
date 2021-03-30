
#!/usr/bin/env python
: '
 Bash script to run python main train or python main test
 Liver Lesion detection model and LIRADs classificaion
arg parse train or test
'

: '
 find process like this ps ax | grep main.py
 kill process like this kill PID
 pgrep -af python
'
nohup python main.py train &
nohup python main.py train > nohup.out &
nohup python main.py train > nohup.log &