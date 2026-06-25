#string
print("******* string Data type *******")
print("Single line string")
print("Hello, World")
print("¯\_(ツ)_/¯")
print("2.99")

print("This is a multi-line string. It can span multiple lines and preserve the formatting.")
print("""Hello, World!
      It's great to be here!""")

print("******* type Data type *******")
print("type of Andrew is", type("Andrew"))
print("type of multi-line string is", type("""Hello, World!
      It's great to be here!"""))
print("type of 2.99 is", type("2.99"))
print("type of 100 is", type(100))
print("type of 3.2 is", type(3.2))

print("******* Python as a calculator! *******")
print(28+35+43+50+65+70+68+66+75+80+95)

print("Complete with chatbot code")
print("Order of operations")
print(75 - 32 * 5 / 9)
print((75 - 32) * 5 / 9)


print("******* Displaying text and calculations together *******")
print("The temperature 75F in degrees Celsius is ((75 - 32) * 5 / 9)C")
print(f"The temperature 75F in degrees Celsius is {((75 - 32) * 5 / 9)}C")
print("Isabel is 28/7 dog years old.")
print(f"Isabel is {28/7} dog years old.")
print(f"Isabel is {28/7:.2f} dog years old.")  # Displaying with 2 decimal places
print(f"Isabel is {28/7:.0f} dog years old.")
print(f"""
    Most countries use the metric system for recipe measurement, 
    but American bakers use a different system. For example, they use 
    fluid ounces to measure liquids instead of milliliters (ml).
    
    So you need to convert recipe units to your local measuring system!
    
    For example, 8 fluid ounces of milk is {8 * 29.5735} ml.
    And 100ml of water is {100 / 29.5735} fluid ounces.
""")

