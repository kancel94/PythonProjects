import stdio
import stopwatch


class EditDistance:
	# return the penalty for aligning char a and char b
	@staticmethod
	def penalty(a, b):
		if a == b:
			return 0
		elif a != b and (a != '-' or b != '-'):
			return 1

	# return the min of 3 integers a,b,c
	@staticmethod
	def min(a, b, c):
		if a < b and a < c:
			return a
		elif b < a and b < c:
			return b
		elif c < a and c < b:
			return c
		elif (a == b and a < c) or (a == c and a < b):
			return a
		elif b == c and b < a:
			return b
		elif a == b and a ==c:
			return a

	# read 2 strings from standard input. (using stdio.readString())
    # compute and print the edit distance between them.
    # output an optimal alignment and associated penalties. 
    # This can be done by retracing steps from opt[0]0] 
    # and checking which of (i+1,j+1), (i, j+1), (i+1, j) 
    # was used as the min from the dynamic programming formula.
	@staticmethod
	def main(): 
		watch=stopwatch.Stopwatch()
		first = stdio.readString()
		second = stdio.readString()
		firstArr = []
		secondArr = []
		firstArr += first
		secondArr += second
		M = len(firstArr)
		N = len(secondArr)
		firstArr += '-'
		secondArr += '-'
		opt= [[0 for i in range(N+1)] for j in range(M+1)]
		for i in range(M):
			for j in range(N):
				opt[i][N] = 2*(M-i)
				opt[M][j] = 2*(N-j)
		for i in range(M-1,-1,-1):
			for j in range(N-1,-1,-1):
				penalty = EditDistance.penalty(firstArr[i], secondArr[j])
				newA = opt[i+1][j+1] + penalty
				#stdio.writeln("newA " + str(newA))
				newB = opt[i+1][j] + 2
				#stdio.writeln("newB " + str(newB))
				newC = opt[i][j+1] + 2
				#stdio.writeln("newC " + str(newC))
				minimum = EditDistance.min(newA, newB, newC)
				opt[i][j] = minimum
		#stdio.writeln(opt)
		stdio.writeln("Edit Distance: " + str(opt[0][0]))
		i = 0
		j = 0
		while i < M and j < N:
			penalty = EditDistance.penalty(firstArr[i], secondArr[j])
			if opt[i][j] == opt[i+1][j+1] + penalty:
				stdio.writeln(str(firstArr[i]) + " " + str(secondArr[j]) + " " + str(penalty))
				i += 1
				j += 1
			elif opt[i][j] == opt[i+1][j] + 2:
				stdio.writeln(str(firstArr[i]) + " - 2")
				i += 1
			elif opt[i][j] == opt[i][j+1] + 2:
				stdio.writeln("- " + str(firstArr[i]) + " 2")
				j += 1
		#stdio.writeln(watch.elapsedTime())


if __name__ == '__main__':
	EditDistance.main()