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
#define LOW_MERGE_CUTOFF 0 // minimum frequency to merge
#define ll long long

int GLOBAL_TOKEN_ID;
map<pair<int, int>, int> MERGE_EVENTS;

struct TrieNode {
    unordered_map<int, pair<int, TrieNode*>> children;
    int mark;
    TrieNode(int mark = -1) {
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

    TrieNode* insert_vector(const vector<int>& tokens, int pos, int lim, int wt, int leaf_mark = -1) {
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
        temp->mark = leaf_mark;
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
        if (bytes.size() != 0)
            frequencies[bytes] += 1;
        
        idx = j+1;
    }
    cout << endl;
    return frequencies;
}

void train(map<vector<int>, int>& corpus, unsigned int vocab_size) {
    /**
    * Optimize BPE tokenization naively.
    */
    GLOBAL_TOKEN_ID = initial_vocab.size();
    // cout << "HEY" << endl;

    while (GLOBAL_TOKEN_ID < vocab_size) {
        int best_x = -1;
        int best_y = -1;
        int best_score = -1;
        // test: the score function is now freq * (vocab[x].size() + vocab[y].size() - 1)
        // int longest_size = 0;
        for (auto& p : corpus) {
            // cout << "p = " << p.first.size() << endl;
            vector<int> word = p.first;
            int freq = p.second;
            for (int i = 0; i + 1 < word.size(); ++i) {
                int x = word[i];
                int y = word[i + 1];
                if (vocab[x].size() + vocab[y].size() > MAX_TOKEN) continue;
                int score = freq * (vocab[x].size() + vocab[y].size() - 1);
                // if (freq > best_freq || (freq == best_freq && vocab[x].size() + vocab[y].size() > longest_size)) {
                if (score > best_score) {
                    best_score = score;
                    best_x = x;
                    best_y = y;
                    // longest_size = vocab[x].size() + vocab[y].size();
                }
            }
        }
        // cout << "Best merge: " << vocab[best_x] << " " << vocab[best_y] << " " << best_freq << endl;
        if (best_score < LOW_MERGE_CUTOFF) break;
        if (best_x == -1) break;
        string new_token = vocab[best_x] + vocab[best_y];
        vocab.push_back(new_token);

        // Update corpus
        map<vector<int>, int> new_corpus;
        for (auto& p : corpus) {
            vector<int> word = p.first;
            int freq = p.second;
            vector<int> new_word;
            int idx = 0;
            while (idx < word.size()) {
                if (idx + 1 < word.size() && (vocab[word[idx]] + vocab[word[idx + 1]] == new_token)) {
                    new_word.push_back(GLOBAL_TOKEN_ID);
                    idx += 2;
                } else {
                    new_word.push_back(word[idx]);
                    idx += 1;
                }
            }
            new_corpus[new_word] += freq;
        }
        GLOBAL_TOKEN_ID += 1;
        corpus = new_corpus;
        cout << "Training: " << GLOBAL_TOKEN_ID << " of " << vocab_size << " tokens,score = " << best_score << " new_token = " << new_token << "                   \r";
    }
    cout << endl;
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
    /**
     * rowechen_ptr goes ahead and searches down the tree.
     * isaac_ptr is more careful and only increases when he's sure that the next token is in the vocabulary.
     * rowechen_ptr will end up running into a lot of dead ends, and ask isaac for help.
    */
    vector<int> tokenized_corpus;
    int rowechen_ptr = 0;
    int isaac_ptr = 0;
    int isaac_notes = 0;
    TrieNode* cur = vocab_trie;
    while(true){
        int chr;
        if(rowechen_ptr >= corpus.size())
            chr = -1;
        else
            chr = corpus[rowechen_ptr]; // New character!
        if (cur->children.find(chr) == cur->children.end()) {
            // Wait a minute, I can't find it in my children!
            // Ask Isaac for help.
            tokenized_corpus.push_back(isaac_notes);
            rowechen_ptr = isaac_ptr + 1;
            if(rowechen_ptr >= corpus.size()) break; // We're done!

            // Start over from the beginning of the trie.
            chr = corpus[rowechen_ptr];
            cur = vocab_trie->children[chr].second;
        } else {
            cur = cur->children[chr].second;
        }
        if(cur->mark != -1){
            isaac_ptr = rowechen_ptr;
            isaac_notes = cur->mark;
        }
        rowechen_ptr++;
        if (rowechen_ptr & 0xfff == 0) cout << "Tokenizing ........ " << rowechen_ptr << " of " << corpus.size() << "\r";
    }
    cout << "Finished tokenizing                             " << endl;
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
    // Maximum allowable number of tokens
    int vocab_size;

    // Load the corpus
    string input_corpus;

    string WHICH_CORPUS = "mahabharata";
    stringstream buf;
    
    if (WHICH_CORPUS == "communistmanifesto") {
        ifstream fin1("..\\..\\corpus\\communistmanifesto.txt");
        buf << fin1.rdbuf();
        fin1.close();
        vocab_size = 1000;
    } else if (WHICH_CORPUS == "mahabharata") {
        ifstream fin1("..\\..\\corpus\\mahabharata1.txt");
        ifstream fin2("..\\..\\corpus\\mahabharata2.txt");
        ifstream fin3("..\\..\\corpus\\mahabharata3.txt");
        buf << fin1.rdbuf() << fin2.rdbuf() << fin3.rdbuf();
        fin1.close();
        fin2.close();
        fin3.close();
        vocab_size = 4000;
    }
    input_corpus = buf.str();
    
    // print corpus length
    cout << "Corpus length: " << input_corpus.size() << endl;

    // Training
    tie(initial_vocab, reverse_initial_vocab) = generate_initial_vocab();
    vocab = initial_vocab;
    map<vector<int>, int> frequencies = pre_tokenize(input_corpus, reverse_initial_vocab);
    train(frequencies, vocab_size);
    // cout << "HEY" << endl;

    // "..//tokenizer_outputs//communistmanifesto_size" + to_string(vocab_size) + "_cap" + to_string(MAX_TOKEN) + ".txt"
    string savefile = "..//tokenizer_outputs//" + WHICH_CORPUS + "_size" + to_string(vocab_size) + "_cap" + to_string(MAX_TOKEN) + ".txt";
    dump_vocab_to_file(savefile);
    cout << "Saved tokens to file: " << savefile << endl;

    // Tokenize text
    cout << "Beginning tokenize" << endl;
    TrieNode* vocab_trie = create_vocab_trie();
    vector<int> ints_corpus = tokenize(input_corpus, vocab_trie);
    cout << "Saving tokenization" << endl;
    dump_tokenization_to_file("mahabharata.tkz", ints_corpus);

    // Tests
    string output_corpus = detokenize(ints_corpus);
    cout << "Compression ratio:" << (double)ints_corpus.size() / (double)input_corpus.size() << endl;

    assert (input_corpus == output_corpus);

    // Debugging tests:
    if(true){        
        vector<string> tests;
        tests.push_back("Hello everyone! This is a test. I hope it works.");
        tests.push_back("x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}");
        for(string test : tests){
            vector<int> ints = tokenize(test, vocab_trie);
            string output = detokenize(ints);
            cout << "Input: " << test << endl;
            cout << "Output: " << output << endl;
            cout << "Compression ratio: " << (double)ints.size() / (double)test.size() << endl;
            cout << endl;
            assert (test == output);
        }
    }    


    // print the distribution of token lengths.
    vector<int> token_lengths(MAX_TOKEN + 1, 0);
    for (string token : vocab) {
        token_lengths[token.size()] += 1;
        assert (token.size() <= MAX_TOKEN);
    }
    for (int i = 0; i <= MAX_TOKEN; ++i) {
        cout << "Token length " << i << ": " << token_lengths[i] << endl;
    }

    // dump_vocab_to_file("communistmanifesto_size4000_cap10.txt");
    return 0;
}
