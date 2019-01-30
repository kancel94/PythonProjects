import sys
import stdio

class RingBuffer:

	#constructor
	def __init__(self, capacity):  # create an empty ring buffer, with given max capacity
		self._cap = capacity
		self._first = 0
		self._last = 0
		self._size = 0
		self._ring = [0]*capacity

	def __str__(self):
		out = ""
		for num in self._ring:
			out+=str(num)+" "
		return out

	def size(self):                # return number of items currently in the buffer
		return self._size

	def isEmpty(self):             # is the buffer empty (size equals zero)?
		return self._size == 0

	def isFull(self):              # is the buffer full  (size equals capacity)?
		return self._size == self._cap

	def enqueue(self, x):          # add item x to the end
		if self.isFull():
			raise Exception('RuntimeException')
		if self._last < len(self._ring):
			if self.isEmpty():
				self._first = self._last
				self._ring[self._last] = x
				self._last += 1
			else:
				self._ring[self._last] = x
				self._last += 1
			self._size += 1
		elif self._last >= len(self._ring):
			self._last = 0
			self._ring[self._last] = x
			self._size += 1
			self._last += 1
		if self._first >= len(self._ring):
			self._first = 0
		#return self._ring, self._size
		
	def dequeue(self):             # delete and return item from the front
		if self.isEmpty():
			raise Exception('RuntimeException')
		self._ring[self._first] = 0
		self._first = (self._first + 1)%len(self._ring)
		self._size -= 1
		#return self._ring, self._size

	def peek(self):                # return (but do not delete) item from the front
		if self.isEmpty():
			raise Exception('RuntimeException')
		return self._ring[self._first]

#for testing
"""def main():
    capacity = int(sys.argv[1])
    ring = RingBuffer(capacity)
    stdio.writeln(ring)
    #stdio.writeln(ring.size())
    #stdio.writeln(ring.isEmpty())
    #stdio.writeln(ring.isFull())
    stdio.writeln(ring.enqueue(1))
    stdio.writeln(ring.enqueue(2))
    stdio.writeln(ring.enqueue(3))
    stdio.writeln(ring.enqueue(4))
    #stdio.writeln(ring.enqueue(5))
    stdio.writeln(ring.dequeue())
    stdio.writeln(ring.dequeue())
    #stdio.writeln(ring.enqueue(10))
    #stdio.writeln(ring.peek())
    #stdio.writeln(ring.size())

if __name__ == '__main__':
    main()"""