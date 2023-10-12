#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <map>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <tuple>
#include <string>
#include <string_view>
#include <queue>


using namespace std;

vector<string> initial_vocab;
map<string, int> reverse_initial_vocab;
// vector<int> initial_vocab_lengths;
vector<string> vocab;

#define MAX_TOKEN 10 // maximum token length
#define LOW_MERGE_CUTOFF 500 // minimum frequency to merge
#define ll long long

int GLOBAL_TOKEN_ID;
map<pair<int, int>, int> MERGE_EVENTS;

struct TrieNode {
    unordered_map<int, pair<int, TrieNode*>> children;
    int mark;
    TrieNode(int mark = 0) {
        // children is a map from token to pair<frequency, TrieNode*>
        children = unordered_map<int, pair<int, TrieNode*>>();
        // If this TrieNode is being used as a tokenizer (as opposed to a corpus)
        // We mark it with an ID corresponding to its token in the vocabulary.
        this->mark = mark;
    }

    ~TrieNode() {
        // Destructor: delete all children.
        for (auto& c1 : children) delete c1.second.second;
    }

    pair<int, int> highest_bytepair_and_pullup() {
        /**
        * Compute the highest frequency length 2 path from this node and pullup.
        * 
        * (best_frequency, best_pair) = max{
        *  (c1->children[c2]->frequency, (c1, c2)) for c1, c2 in this->children, c1->children
        * }
        * 
        */
        pair<int, int> best_pair;
        int best_frequency = -1;
        for (auto& c1 : this->children) {
            int tk1 = c1.first;
            for (auto& c2 : c1.second.second->children) {
                int tk2 = c2.first;
                if (c2.second.first > best_frequency ||
                    (c2.second.first == best_frequency &&
                    vocab[best_pair.first].size() + vocab[best_pair.second].size() < vocab[tk1].size() + vocab[tk2].size()
                    )) {
                    best_frequency = c2.second.first;
                    best_pair = {tk1, tk2};
                }
            }
        }
        if (best_frequency == -1) return {-1, -1};
        // Push this best_pair to the global merge events ''queue''
        MERGE_EVENTS[best_pair] = GLOBAL_TOKEN_ID++;

        // Execute all pending pullup events.
        detect_and_pullup_children();
        for (auto& c1 : this->children)
            c1.second.second->detect_and_pullup_children();
        return best_pair;
    }

    TrieNode* detect_and_pullup_children() {
        /**
        * Detect and execute all pullup events from this node.
        * 
        * for(c1, c2 in this->children, c1->children):
        *   if (c1, c2) in MERGE_EVENTS:
        *     pull_up(c1, c2, MERGE_EVENTS[(c1, c2)])
        */

        vector<pair<pair<int, int>, int>> pull_up_calls;
        for (auto& c1 : this->children) {
            int tk1 = c1.first;
            for (auto& c2 : c1.second.second->children) {
                int tk2 = c2.first;
                if (MERGE_EVENTS.find({tk1, tk2}) != MERGE_EVENTS.end())
                    pull_up_calls.push_back({{tk1, tk2}, MERGE_EVENTS[{tk1, tk2}]});
            }
        }
        for (auto& args : pull_up_calls) {
            this->pull_up(args.first.first, args.first.second, args.second);
        }
        return this;
    }

    TrieNode* pull_up(int tk1, int tk2, int trg) {
        /**
        * Pull up a node if possible.
        * Before:
        *        this
        *         |
        *        tk1
        *       /   \
        *     tk2   etc
        * After:
        *       this
        *      /    \
        *    trg    tk1
        *            |
        *           etc
        * where trg contains the tree originally rooted at tk2.
        */
        TrieNode* c1 = this->children[tk1].second;
        if (c1->children.find(tk2) == c1->children.end()) return nullptr; // This shouldn't happen
        this->children[trg] = c1->children[tk2];
        c1->children.erase(tk2);
        return this;
    }

    TrieNode* insert_vector(const vector<int>& tokens, int pos, int lim, int wt, int leaf_mark = 0) {
        /**
        * Insert the given indices from a vector from [pos, pos + lim) with weight wt.
        */
        TrieNode* temp = this;
        for (int i = pos; i < min((int)tokens.size(), pos + lim); i++) {
            int t = tokens[i];
            if (temp->children.find(t) == temp->children.end()){
                // No child with this token exists. Create one.
                temp->children[t] = {wt, new TrieNode()};
            }
            else{
                // Child with this token exists. Increment its weight.
                temp->children[t].first += wt;
            }
            temp = temp->children[t].second;
        }
        // 
        temp->mark = leaf_mark;
        return this;
    }

    TrieNode* insert_suffixes(const vector<int>& tokens, int lim, int wt) {
        /**
        * Insert all clipped suffixes of the given word with weight wt.
        */
        for (int i = 0; i < tokens.size(); i++) this->insert_vector(tokens, i, lim, wt);
        return this;
    }
};

pair<vector<string>, map<string, int>> generate_initial_vocab() {
    /**
     * This method should generate the initial vocabulary.
     * The initial vocabulary should contain all single characters (bytes) and the <unk> token.
    */
    vector<string> initial_vocab;
    map<string, int> reverse_initial_vocab;
    initial_vocab.push_back("<unk>");
    reverse_initial_vocab["<unk>"] = 0;
    for (int i = 0; i < 256; ++i) {
        string s(1, (char)i);
        initial_vocab.push_back(s);
        reverse_initial_vocab[s] = i + 1;
    }
    // set initial_vocab_lengths to all 1's
    return make_pair(initial_vocab, reverse_initial_vocab);
}

map<vector<int>, int> pre_tokenize(string& corpus, map<string, int> reverse_initial_vocab) {
    /**
     * This method should preform pre-tokenization.
     * First, the corpus should be split by whitespace. Each distinct word should then be mapped to its frequency.
     * Then, each letter in each word should be mapped to its byte (the base vocabulary is 256).
    */
    map<vector<int>, int> frequencies; // maps each distinct word to its frequency.
    int idx = 0;
    while(idx < corpus.size()){
        if (idx % 1000 == 0)
            cout << "Pre-tokenizing: " << idx << " of " << corpus.size() << " frequency size is " << frequencies.size() << "                   \r";
        int j = idx;

        while (j < corpus.size() && corpus[j] != ' ' && corpus[j] != '\n' && corpus[j] != '\t') {
            j += 1;
        }
        
        vector<int> bytes;
        for (int _ = idx; _ < j; _++) {
            char c = corpus.at(_);
            if (reverse_initial_vocab.find(string(1, c)) == reverse_initial_vocab.end()) {
                bytes.push_back(reverse_initial_vocab["<unk>"]);
            } else {
                bytes.push_back(reverse_initial_vocab[string(1, c)]);
            }
        }

        frequencies[bytes] += 1;
        
        idx = j+1;
    }
    cout << endl;
    return frequencies;
}

void trie_train(map<vector<int>, int>& corpus, unsigned int vocab_size) {
    /**
    * Optimize BPE tokenization with a false suffix trie.
    */
    GLOBAL_TOKEN_ID = initial_vocab.size();
    MERGE_EVENTS = map<pair<int, int>, int>();
    TrieNode* trie = new TrieNode();
    int progress = 0;
    int tot = corpus.size();
    for (auto& word : corpus) {
        if (progress % 1000 == 0)
            cout << "Creating trie: inserted " << progress << " of " << tot << "                   \r";
        trie->insert_suffixes(word.first, MAX_TOKEN, word.second);
        ++progress;
    }
    cout << "Completed trie insertion                                                 " << endl;
    while (GLOBAL_TOKEN_ID < vocab_size) {
        pair<int, int> tuple = trie->highest_bytepair_and_pullup();
        if (tuple.first == -1) {
            cout << "No more merges possible.                                         " << endl;
            return;
        }
        vocab.push_back(vocab[tuple.first] + vocab[tuple.second]);
        cout << "Current vocab size=" << vocab.size() << " created token " << vocab.back() << "                   \r";
    }
    cout << "Completed tokenization                                                   " << endl;
}

TrieNode* create_vocab_trie() {
    /**
    * For use in tokenization
    */
    TrieNode* vocab_trie = new TrieNode();
    for (int idx = 0; idx < vocab.size(); idx++) {
        string tk = vocab[idx];
        vector<int> vectorize;
        for (int c : tk) vectorize.push_back(c);
        vocab_trie->insert_vector(vectorize, 0, MAX_TOKEN, 1, idx);
    }
    return vocab_trie;
}

vector<int> tokenize(string& corpus, TrieNode* vocab_trie) {
    vector<int> tokenized_corpus;
    int ptr;
    TrieNode* cur;
    for (ptr = 0, cur = vocab_trie; ptr < corpus.size(); ++ptr) {
        int chr = corpus[ptr];
        if (cur->children.find(chr) == cur->children.end()) {
            tokenized_corpus.push_back(cur->mark);
            cur = vocab_trie->children[chr].second;
        } else
            cur = cur->children[chr].second;
        if (ptr & 0xfff == 0) cout << "Tokenizing ........ " << ptr << " of " << corpus.size() << "\r";
    }
    cout << "Finished tokenizing                             " << endl;
    if (cur != vocab_trie)
        tokenized_corpus.push_back(cur->mark);
    return tokenized_corpus;
}

string detokenize(vector<int>& tokenized_corpus) {
    string corpus = "";
    for (int token : tokenized_corpus)
        corpus += vocab[token];
    return corpus;
}

void cout_dump() {
    for (auto tk : vocab) cout << tk << " ";
    cout << endl;
}

void dump_vocab_to_file(string vocab_file) {
    ofstream fout(vocab_file);
    for (string token : vocab) {
        assert(token.size() <= MAX_TOKEN);
        fout << "<BRUH>" << token;
    }
    fout.close();
}

void dump_tokenization_to_file(string tokenized_file, vector<int> tokens) {
    ofstream fout(tokenized_file);
    for (int i : tokens) fout << i << " ";
    fout << endl;
    fout.close();
}

int main() {
    cout << "HI" << endl;
    // Maximum allowable number of tokens
    int vocab_size = 4000;

    // Load the corpus
    string input_corpus;

    string WHICH_CORPUS = "mahabharata";
    stringstream buf;
    
    if (WHICH_CORPUS == "communistmanifesto") {
        ifstream fin1("..\\..\\communistmanifesto.txt");
        buf << fin1.rdbuf();
        fin1.close();
    } else if (WHICH_CORPUS == "mahabharata") {
        ifstream fin1("..\\..\\corpus\\mahabharata1.txt");
        ifstream fin2("..\\..\\corpus\\mahabharata2.txt");
        ifstream fin3("..\\..\\corpus\\mahabharata3.txt");
        buf << fin1.rdbuf() << fin2.rdbuf() << fin3.rdbuf();
        fin1.close();
        fin2.close();
        fin3.close();
    }
    input_corpus = buf.str();
    
    // print corpus length
    cout << "Corpus length: " << input_corpus.size() << endl;

    // Training
    tie(initial_vocab, reverse_initial_vocab) = generate_initial_vocab();
    vocab = initial_vocab;
    map<vector<int>, int> frequencies = pre_tokenize(input_corpus, reverse_initial_vocab);
    trie_train(frequencies, vocab_size);
    
    // Save the tokens
    if (WHICH_CORPUS == "communistmanifesto")
        dump_vocab_to_file("communistmanifesto_size" + to_string(vocab_size) + "_cap" + to_string(MAX_TOKEN) + ".txt");
    else if (WHICH_CORPUS == "mahabharata")
        dump_vocab_to_file("mahabharata_size" + to_string(vocab_size) + "_cap" + to_string(MAX_TOKEN) + ".txt");

    // Tokenize text
    cout << "Beginning tokenize" << endl;
    vector<int> ints_corpus = tokenize(input_corpus, create_vocab_trie());
    cout << "Saving tokenization" << endl;
    dump_tokenization_to_file("mahabharata.tkz", ints_corpus);

    // Tests
    string output_corpus = detokenize(ints_corpus);
    cout << "Compression ratio:" << (double)ints_corpus.size() / (double)input_corpus.size() << endl;

    // verify that the corpus is the same after detokenization.
    assert(input_corpus == output_corpus);

    // print the distribution of token lengths.
    vector<int> token_lengths(MAX_TOKEN, 0);
    for (string token : vocab) {
        token_lengths[token.size()] += 1;
    }
    for (int i = 0; i < MAX_TOKEN; ++i) {
        cout << "Token length " << i << ": " << token_lengths[i] << endl;
    }

    // dump_vocab_to_file("communistmanifesto_size4000_cap10.txt");
    return 0;
}
