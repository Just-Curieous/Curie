import tool

command = ''' 
cat << EOT > junktest.sh
CPU_UTILIZATION=$(ssh -o StrictHostKeyChecking=no -i $KEY_PATH ec2-user@$PUBLIC_IP 'top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
EOT
'''

print(tool.execute_shell_command(command))