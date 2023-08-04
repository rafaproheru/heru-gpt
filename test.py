import guidance
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("text-davinci-003")

# define a guidance program that adapts proverbs
program = guidance("""Tweak this proverb to apply to model instructions instead.

{{proverb}}
- {{book}} {{chapter}}:{{verse}}

UPDATED
Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
- GPT {{gen 'chapter'}}:{{gen 'verse'}}""")

# execute the program on a specific proverb
executed_program = program(
    proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
    book="Proverbs",
    chapter=11,
    verse=14
)

# Tweak this proverb to apply to model instructions instead.

# Where there is no guidance, a people falls,
# but in an abundance of counselors there is safety.
# - Proverbs 11:14

# UPDATED
# Where there is no guidance, a model fails,
# but in an abundance of instructions there is safety.
# - GPT  11:14

print(executed_program["rewrite"]
)

# we can also get all the program variables as a dictionary
print(executed_program.variables())