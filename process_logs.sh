for i in $(ls $1/*.json)
do
    echo $i
    python3 report_best.py $i
    echo
done