#!/usr/bin/env python
# coding: utf-8

# # Programming Assignment: 🍭 Candy Analysis 📊

# Welcome to the last programming assignment! 🎉 You've been tasked by "DeepLearning.AI Sugar Rush Delights" 🍬 to perform analysis on their candy catalog. 🍬 You have to dive into some sweet data about different candies, figure out which ones are the most popular, and give the marketing team some insights to sweeten their campaigns. 💪 Let's get started!

# ## Before you begin 🚦

# If you **need a fresh copy of the assignment (reset)**, follow the provided [instructions](https://community.deeplearning.ai/t/downloading-your-notebook-downloading-your-workspace-and-refreshing-your-workspace/475495).
# 
# <h2 style="color:green; font-weight:bold;">INSTRUCTIONS FOR ATTEMPTING THE ASSIGNMENT:</h2>
# 
# - **Work exclusively in the notebook** provided to you, named `C1M4_Assignment.ipynb`. **DO NOT** create a new notebook or rename any other file to this default name, **as doing so will break the grading system, causing your tests and submission to fail**.
# 
# 
# - Before starting each exercise, read the instructions carefully. Look for the part called **Your Task**, it tells you exactly what you need to code and gives you all the details you need.
# 
# 
# - In each exercise cell, look for comments `### START CODE HERE ###` and `### END CODE HERE ###`. These show you where to write your code. **Do not add or change any code that is outside these comments, or add any extra code cells in the notebook.**
# 
# 
# - After you finish coding an exercise, there will be a test section that checks your work using a function called `test_your_code`. If everything is correct, you'll see a message saying "<span style="color: green;">All tests passed!</span>" and you can move on. <span style="color: red;">If there's a mistake, you'll see a red message explaining what went wrong, so you can fix it.</span>
# 
# 
# - **Before submitting your notebook for grading, ensure ALL exercises are complete (gotten All tests passed! for all of them)**. Save your work by clicking the 💾 icon at the top left, then click the <span style="color: blue; font-weight: bold;">Submit assignment</span> button at the top right.
# 
# <h2 style="color:green; font-weight:bold;">ASKING FOR HELP IN THE COMMUNITY FORUM:</h2>
# 
# - Sign up for our [Community Forum](https://community.deeplearning.ai/) if you haven't already. Once signed up, post your questions in the [AP4B category](https://community.deeplearning.ai/c/course-q-a/ai-python-for-beginners/463) with detailed information, such as the item's name, exercise number, or code cell. This helps others understand your issue better. Share screenshots if you can, but **don't post solution code publicly**; instead, share the error message you receive.
# 
# With that out of the way, let's begin!

# ## Table of Exercises
# 
# - [Exercise 1: Unwrapping the Data 🎁](#1)
# - [Exercise 2: Popularity Contest! 🏆](#2)
# - [Exercise 3: Finding the Average Sweet Spot 🎯](#3)
# - [Exercise 4: The Top Candy Leaderboard 🥇](#4)
# - [Exercise 5: Crafting the Perfect Description ✍️](#5)

# ### Assignment Starts From Here
# 
# Before starting the assignment, run the cell below. It will bring in some helpful code to check if your solutions are correct and provide feedback if you need to make changes.
# 
# **IMPORTANT NOTE**:  Always run this cell when starting or resuming your assignment. **DO NOT include it in any other notebook cells.**

# In[ ]:


import test_your_code


# <a name='1'></a>
# ## Exercise 1: Unwrapping the Data 🎁
# 
# **Instructions:**
# 
# - The `ex1_helper_functions.py` file includes two functions:
#     -  `read_candy_data`: This function will help you open up the candy data file and see what's inside.  Think of it like opening a box of chocolates! 🍫
#     - `display_table`: This handy function takes the candy data you've read and displays it neatly in a table.
#     
# -  Import both of these helpful functions from the `ex1_helper_functions` file.
# 
# **Your task:**
# 
# - Import all (`*`) of the functions from the `ex1_helper_functions`

# In[26]:


### START CODE HERE ###

# Import using the format "from file_name import all (*)"
# Add your code here
from ex1_helper_functions import *

### END CODE HERE ###


# **Note:** Once you see the <span style="color: green;">All tests passed! Correct import statement used!</span> message, you can proceed the next step. Otherwise, it would mean your import statement or its format is not as expected. Fix it based on the feedback and re-run the exercise and test cells.

# In[27]:


# Test your code!
test_your_code.check_ex1_import_statement()


# **Your task:**
# *   Put that `read_candy_data` function to work!  Use it to read the file `candy_data.csv`. 
#     *   Make sure you pass the file name `candy_data.csv` as a string to the function.    

# In[28]:


### START CODE HERE ###

# Use 'read_candy_data' to read the file `candy_data.csv`
candy_data = read_candy_data("candy_data.csv")

### END CODE HERE ###

# Display the contents loaded into the `candy_data` variable
display_table(candy_data)


# #### Expected Output
# 
# | Candy Name            | Popularity Score | Price in USD |
# |-----------------------|------------------|------------------|
# | Twix  | 92  | 1.25               |
# | PayDay     | 83  | 1.00               |
# | .  | .               | .
# | .| .               | .
# | .    | .               | .
# | Rolo  | 84  | 1.00               |
# | Dove Chocolate     | 84  | 1.00               |

# In[29]:


# Test your code!
test_your_code.exercise_1(candy_data)


# <a name='2'></a>
# ## Exercise 2: Popularity Contest! 🏆
# 
# **Instructions:**
# 
# - The `ex2_helper_functions.py` file has more tools for you!
#     - `get_popularity_scores`: This function goes through your candy data and pulls out just the popularity scores. Think of it like separating the cherry filling from the rest of the candy. 🍒
#     -  `print_scores`: This function takes a list of scores and prints them for you.
# 
# **Your task:**
# 
# - Import the two `get_popularity_scores` and `print_scores` functions from `ex2_helper_functions`

# In[30]:


### START CODE HERE ###

# Import using the format "from file_name import function_name_1, function_name_2"
# Add your code here
from ex2_helper_functions import get_popularity_scores, print_scores

### END CODE HERE ###


# **Note:** Once you see the <span style="color: green;">All tests passed! Correct import statement used!</span> message, you can proceed the next step. Otherwise, it would mean your import statement or its format is not as expected. Fix it based on the feedback and re-run the exercise and test cells.

# In[31]:


# Test your code!
test_your_code.check_ex2_import_statement()


# **Your task:**
# *   Now, let's see those popularity scores! Use the `get_popularity_scores` function with the `candy_data` you loaded earlier. 

# In[32]:


### START CODE HERE ###

# Use 'get_popularity_scores' and pass in the lst `candy_data`
popularity_scores = get_popularity_scores(candy_data)

### END CODE HERE ###

# Print the `popularity_scores` list
print_scores(popularity_scores)


# #### Expected Output:
# 
# ```
# [92, 83, 85, 84, 83, 94, 84, 95, 84, 83, 91, 83, 88, 84, 84, 84, 84, 84, 84]
# ```

# In[33]:


# Test your code!
test_your_code.exercise_2(popularity_scores)


# <a name='3'></a>
# ## Exercise 3: Finding the Average Sweet Spot 🎯
# 
# **Instructions:**
# 
# - Python has a special built-in module called `statistics`.  It's like a secret compartment in your candy box full of useful tools! 🧰 
# - You're going to use the `mean` function from the module `statistics` to find the average popularity score.
# 
# **Your task:**
# 
# - Import the Python package `statistics` *as* `stats`

# In[34]:


### START CODE HERE ###

# Import using the format "import module_name as alias"
# Use the alias name as "stats"
# Add your code here
import statistics as stats

### END CODE HERE ###


# **Note:** Once you see the <span style="color: green;">All tests passed! Correct import statement used!</span> message, you can proceed the next step. Otherwise, it would mean your import statement or its format is not as expected. Fix it based on the feedback and re-run the exercise and test cells.

# In[35]:


# Test your code!
test_your_code.check_ex3_import_statement()


# **Your task:**
# 
# * Time to calculate! Use the `mean` function from your `stats` module (remember how to use an alias with the dot `.` ) to find the average popularity score of all the candies.

# In[36]:


### START CODE HERE ###

# Access the "mean" function in "stats"
# Pass "popularity_scores" to the mean function
avg_popularity = stats.mean(popularity_scores)

### END CODE HERE ###

### Print the average upto 2 decimal places
print(f"Average Popularity Score: {avg_popularity:.2f}")


# #### Expected Output
# 
# ```
# Average Popularity Score: 85.95
# ```

# In[13]:


# Test your code!
test_your_code.exercise_3(avg_popularity)


# <a name='4'></a>
# ## Exercise 4: The Top Candy Leaderboard 🥇
# 
# 
# **Instructions:**
# 
# - You have even more tools in your `ex4_helper_functions.py` file!
#     -  `get_top_candies`:  This function is like a talent scout - it will find the candies whose popularity scores are above a certain level, "avg_popularity" (the average popularity you calculated in Exercise 3). 
#     -  `display_pretty_table`:  This one makes the results look extra nice. ✨
# 
# **Your task:**
# 
# - Import the entire file `ex4_helper_functions`

# In[39]:


### START CODE HERE ###

# Import using the format "import file_name"
# Add your code here
import ex4_helper_functions

### END CODE HERE ###


# **Note:** Once you see the <span style="color: green;">All tests passed! Correct import statement used!</span> message, you can proceed the next step. Otherwise, it would mean your import statement or its format is not as expected. Fix it based on the feedback and re-run the exercise and test cells.

# In[40]:


# Test your code!
test_your_code.check_ex4_import_statement()


# **Your task:**
# 
# - Use the `get_top_candies` function to round up all the candies that have a popularity score higher than or equal to the `avg_popularity` you found. 
# 
# <details>
#     <summary><span style="color: blue; font-weight: bold;">Hint: (click here to open)</span></summary>
#     <p>When you have imported the entire module/local file, you can access any function from it using the <code>.</code> (dot) character.</p>
# </details>

# In[48]:


### START CODE HERE ###

# Access the "get_top_candies" function in "ex4_helper_functions"
# Pass "candy_data" and "avg_popularity" to the "get_top_candies" function
top_candies = []

### END CODE HERE ###

### Display the "top_candies" list
ex4_helper_functions.display_pretty_table(top_candies)


# #### Expected Output:
# 
# ```
# +------------+------------------+--------------+
# | Candy Name | Popularity Score | Price in USD |
# +------------+------------------+--------------+
# |    Twix    |        92        |     1.25     |
# |   M&M's    |        94        |     1.25     |
# |  Snickers  |        95        |     1.25     |
# |  Kit Kat   |        91        |     1.25     |
# | Starburst  |        88        |     1.0      |
# +------------+------------------+--------------+
# ```

# In[44]:


# Test your code!
test_your_code.exercise_4(top_candies)


# <a name='5'></a>
# ## Exercise 5: Crafting the Perfect Description ✍️
# 
# Great work identifying the top candies! 🎉 Now, let's help the marketing team really sell these tasty treats. 🍫 You'll use the `get_llm_response` function provided to you below to generate a short, catchy two-sentence description for each of the top candies. 
# 
# You can adjust the temperature and content variables to fine-tune the responses from the LLM.
# 
# Run the cell below to import the `client` which will help run the `get_llm_response` function.

# In[45]:


from ex5_helper_functions import client


# **Your task:**
# - Play around with different `temperature` settings.
# - Play around with a different `content` description.
# 
# **Note** In order to get a successful grade for this exercise, you have to set values for other than `temperature=0.0` and  `"content": "You are an AI assistant."`
# 
# **Fun Suggestion**: For `content`, you can use "You talk like a Pirate." and temperature as `0.5`.

# In[46]:


### START CODE HERE ###

def get_llm_response(prompt):
    completion = client.chat.completions.create( 
        model='gpt-4o-mini', 
        messages=[ 
            {
                "role": "system",
                "content": "You are an Gen AI assistant.", # <-- You have to make change here
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1, # <-- You have to make change here
    )
    response = completion.choices[0].message.content 
    return response

### END CODE HERE ###


# In[47]:


# Test your code!
test_your_code.check_change_in_ex5()


# **Note:** Once you see the <span style="color: green;">All tests passed!</span> message, it means you can move on to the next step. Otherwise, make changes to your `get_llm_response` as instructed.

# ### Submission Note:
# If you have passed all the tests up to this point, you can submit your assignment for grading.
# 
# **But before you submit your assignment, re-run the assignment just in case there are any unexpected errors present in the notebook once it goes for grading.** To do so, follow these steps:
# 
# 1. Restart the `Kernel` and select the `Restart & Clear Output` option. You can do this by clicking on the `Kernel` menu at the top of the notebook.
# ![Kernel Restart Image](kernel_restart.png)
# 2. Once the kernel restarts and all outputs are clear, run the cells from top to bottom again up to this point.
# 
# If you have followed these steps and still pass all of the tests, you can submit your assignment for grading. If you encounter any errors, please fix them before submitting.
# 
# To submit your assignment for grading, save your work by clicking the 💾 icon at the top left, then click the <span style="color: blue; font-weight: bold;">Submit assignment</span> button at the top right. Good luck!
# 
# Running the below cell is OPTIONAL and will not affect your grading in anyway.

# In[ ]:


for candy in top_candies:
    
    prompt = f"""
    For the given candy name, {candy['Candy Name']}, write a short, catchy two-sentence description.
    """
    
    response = get_llm_response(prompt)
    
    print(f"NAME: {candy['Candy Name']}")
    print(f"DESCRIPTION: {response}")
    print()


# ## Conclusion
# That's a wrap! You've used Python to import data, analyze it, and even generate descriptions! Give yourself a pat on the back for completing this assignment. You've taken another step forward in your Python journey, and we can't wait to see what you accomplish next! 😊
