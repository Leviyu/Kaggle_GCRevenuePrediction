#!/bin/tcsh

set max_job = 19


set per = 100000


set job = 0
while($job <= $max_job)
echo "--> on jon $job"

./c01.extract.py $job $per  > & log.${job} &
##python3 ./c01.extract.py $job $per > & log.${job} &
sleep 1s
@ job ++
end 



