n= 100
a=1
b=2
count = 0
for i in range(n // a + 1):
    for j in range(n // b + 1):
        if i * a + j * b == n:
            count += 1

print(count)

