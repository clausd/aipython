An AI improved version of the ipython command line

Basic functionality. Instead of writing python I can ask for an outcome and the REPL will use an AI model to turn this into python instructions. 

I will allow 3-10 instructions per AI instruction. This should be configurable in an .aipython config file. 
Command output is automatically added back to the AI system for context

Also configurable whether to just run python or go into yolo mode and run the commands

Escapes emulate ipython's shell escape wit with an extra ! thrown on so there are three modes

AI mode
! this is now a python command
!! this is a shell command
? is AI questions 

Magic commands still work as per usual so they drop us into python straight away


If possible lean on the history mechanism to store AI instructions and command path. 

Simplest option is probably to include the AI instructions as a comment block and push that to history with an

# Magics

There's an %ai-config magic to control the AI use and mode use
There's an %ai magic... for users who want to keep the ipython logic intact but just want help to write the commands

%ai writes the next instruction (or 5) with AI
%ai? lets you ask a question for the bot

# AI! 
line at the top for commands, and an 

# AI?
for AI questions. 


# Agent perf

The first and immediate goal is to simply be able to describe what I want instead of remembering e.g. complex panda invocations. The agent is able to use history and variable context to understand requests. To save tokens these are available as tools. 

Other tools we want to add are 
* command execution and feeding command output into the context
* mentioning clipboard should work
* A magic to take last command output and copy to clipboard
* Referencing files just downloaded or in clipboard or just moved - I want to be able to say "Describe the sheet I just downloaded"

More will come as we go along - this is already too much to begin with


# How to build it

Down the road we want to be AI agnostic but for now we'll lean hard on Claude through the claudette library (it's convenient for tool definitions and agent loop)

for the REPL control maybe just use what ipython does? Alternately suggest a super snazze version

use the InteractiveShell class for the IPython interaction

