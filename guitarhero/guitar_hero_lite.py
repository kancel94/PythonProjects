
from guitar_string import GuitarString
import stdaudio
import stddrawtim
import stdio

if __name__ == "__main__":
	keyboard = ["q", "2", "w", "e", "4", "r", "5", "t", "y", "7", "u", "8", "i", "9", "o", "p", "-", "[", "=", "z", "x", "d", "c", "f", "v", "g", "b", "n", "j", "m", "k", ",", ".", ";", "/", "'", " "]
	objects = [0 for _ in range(len(keyboard))]
	
	for i in range(len(keyboard)):
		frequency = 440.0*2**((i-24)/12)
		objects[i] = GuitarString(frequency)
	while True:
	# check if the user has typed a key if so, process it  
		sample = 0
		if (stddrawtim.hasNextKeyTyped()):
			key = stddrawtim.nextKeyTyped()
			objects[keyboard.index(key)].pluck()
		for i in range(0, len(keyboard)):
			sample += objects[i].sample()
		stdaudio.playSample(sample)
		for i in range(0, len(keyboard)):
			objects[i].tic()

