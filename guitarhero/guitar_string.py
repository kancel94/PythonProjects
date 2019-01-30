import random
import stdio
from ring_buffer import RingBuffer

class GuitarString:

	def __init__(self, frequency):
		desiredCap = 44100/frequency
		self._N = round(desiredCap)
		self._ringBuffer = RingBuffer(self._N)
		self._counter = 0

	
	def __str__(self):
		return str(self._ringBuffer)

	def pluck(self):                         # set the buffer to white noise
		if self._ringBuffer.isFull():
			for i in range(0,self._N):
				self._ringBuffer.dequeue()
		if self._ringBuffer.isEmpty():
			for i in range(0,self._N):
				val = random.uniform(-1.5, 1.5)
				self._ringBuffer.enqueue(val)

	def tic(self):                           # advance the simulation one time step
		self._counter += 1
		if self._ringBuffer.size() < 2:
			return
		decayFactor = .996
		KSupdate1 = self._ringBuffer.peek()
		self._ringBuffer.dequeue()
		KSupdate2 = self._ringBuffer.peek()
		update = (KSupdate1+KSupdate2)/2*decayFactor
		self._ringBuffer.enqueue(update)
		

	def sample(self):                        # return the current sample
		if self._ringBuffer.isEmpty():
			return 0
		else:
			return self._ringBuffer.peek()
	
	def time(self):                          # return number of tics
		return self._counter

#for testing
"""def main():
	hi=GuitarString(2000)
	stdio.writeln(hi.pluck())
	stdio.writeln(hi.tic())
	stdio.writeln(hi.tic())
	stdio.writeln(hi.sample())
	stdio.writeln(hi.time())

if __name__ == '__main__':
    main()"""