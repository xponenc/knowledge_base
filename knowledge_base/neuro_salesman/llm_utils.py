from langchain_core.runnables import Runnable


class VerboseLLMChain(Runnable):
    def __init__(self, chain: Runnable, chain_name: str, debug_mode: bool = False):
        self.chain = chain
        self.chain_name = chain_name
        self.verbose = debug_mode

    def invoke(self, inputs: dict, config=None, **kwargs):
        if self.verbose:
            print(f"[{self.chain_name}] inputs: {inputs}")
        result = self.chain.invoke(inputs, config=config, **kwargs)
        if self.verbose:
            print(f"[{self.chain_name}] output: {result}")
        return result

    # @property
    # def output_key(self):
    #     return getattr(self.chain, "output_key", None)
