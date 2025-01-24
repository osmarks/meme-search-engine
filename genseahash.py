import seahash, sys

with open(sys.argv[1], "rb") as f:
	print(seahash.hash(f.read()))
