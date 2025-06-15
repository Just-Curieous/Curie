import tool

command = '''
uuid=$(openssl rand -hex 16 | awk '{
    for (i = 1; i <= length($0); i += 2) {
        printf "%s-", substr($0, i, 2)
    }
    printf "\\n"
}' | sed 's/-$//') # Each key needs to be unique, otherwise we don't have access to it and SSH will fail. 
echo $uuid
'''

print(tool.write_to_file(
    {"input_string": command, "file_path": "dump2.sh"}))
