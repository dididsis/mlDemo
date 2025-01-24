import csv
import yaml
import streamlit_authenticator as stauth

users_csv_path = "user_info.csv"
config_yaml_path = "config.yaml"

with open(users_csv_path, "r") as f:
    csvreader = csv.DictReader(f)
    users = list(csvreader)

with open(config_yaml_path,"r") as f:
    yaml_data = yaml.safe_load(f)

users_dict = {}
for user in users:
    user["password"] = stauth.Hasher([user["password"]]).generate()[0]
    tmp_dict = {
        "name" : user["name"],
        "password" : user["password"],
    }
    users_dict[user["id"]] = tmp_dict

yaml_data["credentials"]["usernames"] = users_dict
with open(config_yaml_path, "w") as f:
    yaml.dump(yaml_data, f)
    print("complete")