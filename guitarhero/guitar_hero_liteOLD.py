
from guitar_string import GuitarString
import stdaudio
import stddraw

if __name__ == "__main__":
	# Create two guitar strings, for concert A and C

	CONCERT_A = 440.0
	CONCERT_C = CONCERT_A * (2 ** 3.0/12.0)   
	stringA =  GuitarString(CONCERT_A)
	stringC =  GuitarString(CONCERT_C)

	while True: 
		if (stddraw.hasNextKeyTyped()):
			key = stddraw.nextKeyTyped()
			if key == 'a':
				stringA.pluck() 
	      	#elif key == 'c':
	    		#stringC.pluck()
	  

		# compute the superposition of samples
		sample = stringA.sample() + stringC.sample()

		# play the sample on standard audio
		stdaudio.playSample(sample)

		# advance the simulation of each guitar string by one step   
		stringA.tic()
		stringC.tic()