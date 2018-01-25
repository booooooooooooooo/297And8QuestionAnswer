


with open("/output/a.txt", "w") as f:
    f.write("hello")
with open("/output/a.txt") as f:
    word = f.readline()
print word
