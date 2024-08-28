# Creating a list
my_list = [1, 2, 3, 4, 5]
print("Original list:", my_list)

# Adding an element 
my_list.append(6)
print("List after adding an element:", my_list)

# Removing an element 
my_list.remove(3)
print("List after removing an element:", my_list)

# Modifying an element 
my_list[2] = 10
print("List after modifying an element:", my_list)

# Create a dictionary
my_dict = {"name": "Ahil", "age": 21, "city": "Chennai"}
print("\nOriginal dictionary:", my_dict)

# Adding an element to the dictionary
my_dict["college"] = "SRMV University"
print("Dictionary after adding an element:", my_dict)

# Remove
del my_dict["age"]
print("Dictionary after removing an element:", my_dict)

# Modifying
my_dict["city"] = "Coimbatore"
print("Dictionary after modifying an element:", my_dict)

# Creating 
my_set = {1, 2, 3, 4, 5}
print("\nOriginal set:", my_set)

# Add
my_set.add(6)
print("Set after adding an element:", my_set)

# Removing 
my_set.discard(3)
print("Set after removing an element:", my_set)


my_set.remove(4)
my_set.add(7)
print("Set after modifying elements:", my_set)
