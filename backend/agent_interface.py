from agent import driver, start_agent, query_graph

class AgentInterface():
    def __init__(self):
        pass

    def retrieve_current_graph(self):
        q = '''
        MATCH (N) -[R]- (M)
        RETURN N, R, M;
        '''
        return driver.execute_query(q)
    
    def start_processing(self, filename):
        solution = start_agent(filename)
        return solution
    
    def query_graph(self, user_query):
        return query_graph(user_query)

    
    
