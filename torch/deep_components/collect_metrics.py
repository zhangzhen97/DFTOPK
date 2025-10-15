import sys
import codecs

def parse_metrics(lines):
    metric_names = []
    metric_values = []

    i = 0
    while i < len(lines):
        header_parts = lines[i].strip().split()
        value_parts = lines[i + 1].strip().split()

        if len(header_parts) < 4 or len(value_parts) < 4 or len(header_parts)!=len(value_parts):
            i += 2
            continue

        stage = header_parts[2]
        keys = header_parts[3:]
        values = value_parts[3:]
        if len(keys) != len(values):
            i += 2
            continue

        for k, v in zip(keys, values):
            if "recall" in k.lower() or "ndcg" in k.lower():
                metric_names.append(f"{stage}#{k}")
                metric_values.append("%.4f" %(float(v)))
        i += 2

    header_line = "\t".join(metric_names)
    value_line = "\t".join(metric_values)

    print(header_line)
    print(value_line)

def collect_lines():
    metric_lines = []
    for line in sys.stdin.readlines():
        if not line.startswith("metric"):
            continue
        metric_lines.append(line)
    return metric_lines

def main():
    metric_lines = collect_lines()

    parse_metrics(metric_lines)

if __name__ == "__main__":
    main()
