import time
start = time.time()
print("Hello World")
#time.sleep(2)
num = 10000000
a = 0
for i in range(num):
    a = a +i
end = time.time()
print(end - start)