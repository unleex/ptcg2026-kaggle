import kagglehub

# Download latest version
kagglehub.login()
path = kagglehub.competition_download("pokemon-tcg-ai-battle", output_dir="data")

print("Path to competition files:", path)
