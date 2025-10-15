pwd
METHOD=$1
ls -l logs/TEST_*_$METHOD-1st_E3*S2.log
first=1
for file in logs/TEST_*_$METHOD-1st_E3*_S2.log; do
    if [ "$first" -eq 1 ]; then
        cat "$file" | python deep_components/collect_metrics.py
        first=0
    else
        cat "$file" | python deep_components/collect_metrics.py | tail -n 1
    fi
done
