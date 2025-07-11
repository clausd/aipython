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

There's a new %ai magic which controls config of the ai system like the settings I mention above

If possible lean on the history mechanism to store AI instructions and command path. 

Simplest option is probably to include the AI instructions as a comment block and push that to history with an


# AI! 
line at the top for commands, and an 

# AI?
for AI questions. 

You should use the snazziest REPL library in the python world for this. Syntax highlighting - ability to walk back through history and so on is key. 

# Agent perf

The first and immediate goal is to simply be able to describe what I want instead of remembering e.g. complex panda invocations. The agent is able to use history and variable context to understand requests. To save tokens these are available as tools. 

Other tools we want to add are 
* command execution and feeding command output into the context
* mentioning clipboard should work
* A magic to take last command output and copy to clipboard
* Referencing files just downloaded or in clipboard or just moved - I want to be able to say "Describe the sheet I just downloaded"

More will come as we go along - this is already too much to begin with