from faker import Faker


faker = Faker('zh_CN')
with open("ChineseName.txt", "a", encoding="utf-8") as f:
    for _ in range(50000):
        name = faker.name()
        f.write(f"{name}\n")
