from app.models.generator import Generator


generator = Generator()

answer = generator.generate(
    "What is machine learning? Answer in one clear sentence."
)

print("\nAnswer:")
print(answer)