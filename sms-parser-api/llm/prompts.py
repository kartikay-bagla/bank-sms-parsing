"""
FunctionGemma prompt templates and formatting.
"""

SYSTEM_PROMPT = """You are a financial transaction extractor. Analyze SMS messages and:
1. If the message describes a completed financial transaction (money sent, received, debited, or credited), use extract_transaction to capture the details.
2. If the message is not a transaction (OTP, promotional, application status, payment request, etc.), use skip_message.

Only extract actual completed transactions with concrete amounts, not payment requests or pending transactions."""

TOOLS_DECLARATION = """<start_function_declaration>declaration:extract_transaction{description:<escape>Extract transaction details from a banking SMS message<escape>,parameters:{properties:{source:{type:<escape>STRING<escape>},currency:{type:<escape>STRING<escape>},amount:{type:<escape>NUMBER<escape>},date:{type:<escape>STRING<escape>},destination:{type:<escape>STRING<escape>},type:{type:<escape>STRING<escape>}},required:[<escape>source<escape>,<escape>currency<escape>,<escape>amount<escape>,<escape>date<escape>,<escape>destination<escape>,<escape>type<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration><start_function_declaration>declaration:skip_message{description:<escape>Skip messages that are not financial transactions<escape>,parameters:{properties:{reason:{type:<escape>STRING<escape>}},required:[<escape>reason<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>"""


def format_prompt(sms: str) -> str:
    """Format the prompt for FunctionGemma"""
    return f"""<bos><start_of_turn>developer
{SYSTEM_PROMPT}{TOOLS_DECLARATION}<end_of_turn>
<start_of_turn>user
{sms}<end_of_turn>
<start_of_turn>model
"""
