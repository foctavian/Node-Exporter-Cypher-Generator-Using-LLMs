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
    
    async def start_processing(self, filename):
        solution = await start_agent(filename)
        return solution
    
    def query_graph(self, user_query):
        return query_graph(user_query)

    
    
