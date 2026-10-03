from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv


# Load job roles and skills
roles = []
skills = []

with open("raw_skills.csv", "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        roles.append(row["Role"])
        skills.append(row["Skills"])


# Take user input
print("===== Tech Stack Recommender =====")
print("Enter your 3 skills.")

skill1 = input("Skill 1: ")
skill2 = input("Skill 2: ")
skill3 = input("Skill 3: ")

user_skills = skill1 + " " + skill2 + " " + skill3


# Convert skills into TF-IDF vectors
vectorizer = TfidfVectorizer()

all_data = skills + [user_skills]

vectors = vectorizer.fit_transform(all_data)


# Calculate similarity
user_vector = vectors[-1]
role_vectors = vectors[:-1]

scores = cosine_similarity(user_vector, role_vectors)[0]


# Combine roles with scores
results = list(zip(roles, scores))


# Sort by highest similarity
results.sort(key=lambda x: x[1], reverse=True)


# Show Top 3 recommendations
print("\n===== Recommended Career Roles =====")

for i, (role, score) in enumerate(results[:3], 1):
    print(f"{i}. {role} - Match: {score:.2f}")