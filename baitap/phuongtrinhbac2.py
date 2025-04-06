'''
tinh phuong trinh bac 2
'''
from math import sqrt
a = float(input('nhap tham so a: '))
b = float(input('nhap tham so b: '))
c = float(input('nhap tham so c: '))
delta = b**2 - 4*a*c
if a == 0: #phuong trinh bac nhat
    if b == 0:
        if c == 0: # 0=0
            print('vo so nghiem')
        else: # c = 0
            print('vo nghiem')
    else:
        if c == 0: # bx = 0
            print(0)
        else: # bx + c = 0
            print(-c/b)
else:
    if delta < 0:
        print('vo nghiem')
    elif delta == 0:
        print(c/a)
    elif delta > 0:
        can_delta = sqrt(delta)
        print((-b + can_delta) / 2*a)
        print((-b - can_delta) / 2*a)

            









