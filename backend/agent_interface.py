from agent import driver, start_agent

class AgentInterface():
    def __init__(self):
        pass

    def retrieve_current_graph(self):
        q = '''
        MATCH (N) -[R]- (M)
        RETURN N, R, M;
        '''
        return driver.execute_query(q)
    
    def start_processing(self):
        solution = start_agent()
        return solution

    
    
