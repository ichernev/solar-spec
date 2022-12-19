import csv
import sys

with open(sys.argv[1], 'r') as f:
    for line in csv.reader(f):
        watts, w, h, price, _ = line
        watts, price = map(float, (watts, price))
        w, h = map(int, (w, h))
        area_sqm = w * h / (1000 * 1000)
        print("%.2f lv/w %.2f w/m^2 : %.0fWp %.0f lv [%d %d]" % (
            price / watts, watts / area_sqm, watts, price, w, h))
