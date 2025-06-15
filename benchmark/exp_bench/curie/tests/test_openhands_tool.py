import tool

prompt='''
The starter file can be found under "/starter_file/large_language_monkeys".
Note: there is no need to modify the source code of any files outside "/starter_file/large_language_monkeys". Though you may choose to read the source code just to enhance your understanding of its inner workings.

Setup the python environment for llmonk following the large_language_monkeys/README.md. 

Clean up the files under ./logs. 

OpenAI Azure credentials are available under large_language_monkeys/env.sh

You can verify the success of the setup by running the following command: python llmonk/generate/gsm8k.py

Provide a reproducible experimental workflow named experimental_workflow.sh, which contains the steps to setup the environment.
'''

prompt='''
The starter file can be found under "/workspace".
write a quick sort function in a new file called 'sort.py' under the starter file directory
'''
res = tool.codeagent_openhands(prompt)
print("===========================================")
print(res) 