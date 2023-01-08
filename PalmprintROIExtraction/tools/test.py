import gzip

s = "dsa5d1a1d56a15d6a15ds51das165das165d"

result = gzip.compress(s.encode())
print(len(s), len(result))
