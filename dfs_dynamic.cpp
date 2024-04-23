#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int v, int w) {
        adj[v].push_back(w);
    }

    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);
        parallelDFSUtil(startVertex, visited);
    }

    void parallelDFSUtil(int v, vector<bool>& visited) {
        #pragma omp critical
        {
            if (!visited[v]) {
                visited[v] = true;
                cout << v << " ";
            }
        }

        #pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i) {
            if (!visited[adj[v][i]])
                parallelDFSUtil(adj[v][i], visited);
        }
    }
};


int main() {
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
    cout << "Enter the start vertex for DFS: ";
    cin >> startVertex;

    // Perform DFS and BFS
    cout << "Depth-First Search (DFS): ";
    g.parallelDFS(startVertex);
    cout << endl;

    return 0;
}


