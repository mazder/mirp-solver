import json

# Sample data to serialize
data = {
    "name": "SimpleITK",
    "version": "2.1",
    "dependencies": ["ITK", "GTest"]
}

# Serialize data to a JSON string
json_str = json.dumps(data, indent=4)
print("Serialized JSON:")
print(json_str)

# Deserialize back into a Python object
data_back = json.loads(json_str)
print("\nDeserialized JSON:")
print(data_back)
