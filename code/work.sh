#!/usr/bin/tcsh


set PWD = `pwd`
set ID = $1
echo $ID

if($ID == "" ) then
	echo "Please specify ID"
exit 0
endif

set log = $PWD/LOG/logfile.${ID}
python3 $PWD/run.py  $ID > & $log &
