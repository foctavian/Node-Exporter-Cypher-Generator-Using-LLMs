from agent import driver, start_agent, query_graph, start_update

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
    
    async def query_graph(self, user_query):
        return await query_graph(user_query)
    
    async def start_update(self, ip, name, timestamp):
        return await start_update(ip, name, timestamp)

    
    