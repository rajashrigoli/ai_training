from helper_functions import print_llm_response


print_llm_response("What is capital of France?")

name = "Otto Matic"
dog_age = 21/7
print_llm_response(f"""If {name} were a dog, he would be {dog_age} years old.
Describe what life stage that would be for a dog and what that might 
entail in terms of energy level, interests, and behavior.""")

driver = "unicorn"
drivers_vehicle = "colorful, asymmetric dinosaur car"
favorite_planet = "Pluto"

print_llm_response(f"""Write me a 300 word children's story about a {driver} racing
a {drivers_vehicle} for the {favorite_planet} champion cup.""")

favorite_book = "1001 Ways to Wear a Hat"
second_fav_book = "2002 Ways to Wear a Scarf"
print(f"My most favorite book is {favorite_book}, but I also like {second_fav_book}")


# Make variables for your favorite game, movie, and food.
# Then use print_llm_response to ask the LLM to recommend you
# a new song to listen to based on your likes.

favorite_game = "Cricket"
movie = "The Lord of the Rings"
food = "Pizza"
print_llm_response(f"""I like the game {favorite_game}, the movie {movie}, and the food {food}.
Can you recommend a new song for me to listen to based on my likes?""")