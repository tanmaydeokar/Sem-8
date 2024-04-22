//Q1) Openmp  bfs dfs
#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Graph class representing the adjacency list
class Graph {
    int V;  // Number of vertices
    vector<vector<int>> adj;  // Adjacency list

public:
    Graph(int V) : V(V), adj(V) {}

    // Add an edge to the graph
    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    // Parallel Depth-First Search
    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);
        parallelDFSUtil(startVertex, visited);
    }

    // Parallel DFS utility function
    void parallelDFSUtil(int v, vector<bool>& visited) {
        visited[v] = true;
        cout << v << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i) {
            int n = adj[v][i];
            if (!visited[n])
                parallelDFSUtil(n, visited);
        }
    }

    // Parallel Breadth-First Search
    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        while (!q.empty()) {
            int v = q.front();
            q.pop();
            cout << v << " ";

            #pragma omp parallel for
            for (int i = 0; i < adj[v].size(); ++i) {
                int n = adj[v][i];
                if (!visited[n]) {
                    visited[n] = true;
                    q.push(n);
                }
            }
        }
    }
};

int main() {
    // Example input (commented out)
    /*
    Enter the number of vertices: 7
    Enter the number of edges: 6
    Enter the edges (vertex pairs):
    0 1
    0 2
    1 3
    1 4
    2 5
    2 6
    Enter the start vertex for DFS and BFS: 0
    */

    // Uncomment the following lines to use the example input
    /*
    int numVertices = 7, numEdges = 6;
    Graph g(numVertices);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    g.addEdge(2, 6);
    int startVertex = 0;
    */

    // Or input your own values:
    int numVertices, numEdges;
    cout << "Enter the number of vertices: ";
    cin >> numVertices;
    cout << "Enter the number of edges: ";
    cin >> numEdges;

    // Create a graph
    Graph g(numVertices);

    // Input edges
    cout << "Enter the edges (vertex pairs):" << endl;
    for (int i = 0; i < numEdges; ++i) {
        int v, w;
        cin >> v >> w;
        g.addEdge(v, w);
    }

    // Input start vertex
    int startVertex;
    cout << "Enter the start vertex for DFS and BFS: ";
    cin >> startVertex;

    // Perform DFS and BFS
    cout << "Depth-First Search (DFS): ";
    g.parallelDFS(startVertex);
    cout << endl;

    cout << "Breadth-First Search (BFS): ";
    g.parallelBFS(startVertex);
    cout << endl;

    return 0;
}






//Q2)BFS DFS CUDA
// BFS 
!nvcc --version
!pip install git+https://github.com/afnan47/cuda.git
%load_ext nvcc_plugin


  
%%writefile bfs.cu
 #include <iostream>
 #include <queue>
 #include <vector>
 #include <omp.h>
 using namespace std;

 int main()
 {
 int num_vertices, num_edges, source;
 cout << "Enter number of vertices, edges, and source node: ";
 cin >> num_vertices >> num_edges >> source;
 // Input validation
 if (source < 1 || source > num_vertices) {
 cout << "Invalid source node!" << endl;
 return 1;
 }

vector<vector<int>> adj_list(num_vertices + 1);
  for (int i = 0; i < num_edges; i++)
  {
        int u, v;
        cin >> u >> v;
        // Input validation for edges
        if (u < 1 || u > num_vertices || v < 1 || v > num_vertices)
        {
            cout << "Invalid edge: " << u << " " << v << endl;
            return 1;
        }
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
  }

queue<int> q;
vector<bool> visited(num_vertices + 1, false);
q.push(source);
visited[source] = true;
while (!q.empty())
{
        int curr_vertex = q.front();
        q.pop();
        cout << curr_vertex << " ";
        // Sequential loop for neighbors
        for (int i = 0; i < adj_list[curr_vertex].size(); i++)
        {
            int neighbour = adj_list[curr_vertex][i];
            if (!visited[neighbour])
            {
                visited[neighbour] = true;
                q.push(neighbour);
            }
        }
    }
    cout << endl;
    return 0;
}

!nvcc breadthfirst.cu -o bfs
!./bfs

