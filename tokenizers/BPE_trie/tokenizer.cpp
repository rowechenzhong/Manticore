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

#define NUM_FILES 10 // Number of corpus docs to use
#define MAX_TOKEN 400 // maximum token length
#define LOW_MERGE_CUTOFF 0 // minimum frequency to merge
#define CORPUS_DELIMITER char(30) // delimiter between tokens
#define OUTPUT_DELIMITER char(31) // delimiter between tokens
#define ll long long

#define CORPUS_DIR "./corpus/falcon/train/"
#define VOCAB_OUTPUT "./tokenizers/tokenizer_outputs/peregrine_40k.txt"

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
    int idx = 0;
    initial_vocab.push_back("<unk>");
    reverse_initial_vocab["<unk>"] = idx++;
    for (int i = 0; i < 256; ++i) {
        string s(1, (char)i);
        initial_vocab.push_back(s);
        reverse_initial_vocab[s] = idx++;
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

vector<int> to_ints(string& corpus, map<string, int>& reverse_initial_vocab) {
    vector<int> res;
    int unk_count = 0;
    for (char c : corpus) {
        string s(1, c);
        if (reverse_initial_vocab.find(s) != reverse_initial_vocab.end()) {
            res.push_back(reverse_initial_vocab[s]);
        } else {
            res.push_back(reverse_initial_vocab["<unk>"]);
            unk_count += 1;
        }
    }
    // print unk count
    cout << "Unk count: " << unk_count << endl;
    cout << "Unk ratio: " << (double)unk_count / (double)corpus.size() << endl;
    return res;
}


void train(vector<int>& corpus, unsigned int vocab_size) {
    while (vocab.size() < vocab_size) {
        int batch_size = min((int)(vocab_size - vocab.size()), 400);
        vector<pair<int, int>> max_frequency_length_queue(batch_size, {0, 0});
        vector<pair<int, int>> max_i_j_queue(batch_size, {-1, -1});
        cout << "Current vocab size=" << vocab.size() << " current corpus size=" << corpus.size() << "                        \r";
        unordered_map<int, unordered_map<int, int>> frequencies;
        for (size_t i = 0; i < corpus.size() - 1; ++i) {
            pair<int, int> i_j = {corpus[i], corpus[i+1]};
            pair<int, int> info = {++frequencies[i_j.first][i_j.second], - vocab[i_j.first].size() - vocab[i_j.second].size()};
            for (int j = 0; j < max_frequency_length_queue.size(); j++) {
                if (i_j == max_i_j_queue[j]) {
                    max_frequency_length_queue.erase(max_frequency_length_queue.begin() + j);
                    max_i_j_queue.erase(max_i_j_queue.begin() + j);
                    break;
                }
            }
            for (int j = 0; j < max_i_j_queue.size(); j++) {
                if (info > max_frequency_length_queue[j]) {
                    max_frequency_length_queue.insert(max_frequency_length_queue.begin() + j, info);
                    max_i_j_queue.insert(max_i_j_queue.begin() + j, i_j);
                    if (max_i_j_queue.size() > batch_size) {
                        max_frequency_length_queue.pop_back();
                        max_i_j_queue.pop_back();
                    }
                    break;
                }
            }
            if (max_i_j_queue.size() != batch_size) {
                max_frequency_length_queue.push_back(info);
                max_i_j_queue.push_back(i_j);
            }
        }
        int ouch = vocab.size();
        for (auto i_j : max_i_j_queue)
            vocab.push_back(vocab[i_j.first] + vocab[i_j.second]);

        vector<int> new_corpus;
        size_t ptr = 0;

        while (ptr < corpus.size()) {
            if (ptr < corpus.size() - 1) {
                for (int j = 0; j < batch_size; j++) { 
                    pair<int, int> i_j = max_i_j_queue[j]; 
                    if (corpus[ptr] == i_j.first && corpus[ptr + 1] == i_j.second) {
                        new_corpus.push_back(ouch + j);
                        ptr += 2; 
                        goto end;
                    }
                }
                new_corpus.push_back(corpus[ptr]);
                ptr += 1;
            } else {
                new_corpus.push_back(corpus[ptr]);
                ptr += 1;
            }
            end:;
        }
        corpus = new_corpus;
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
    ofstream fout(vocab_file, ios::binary);
    for (string token : vocab) {
        assert(token.size() <= MAX_TOKEN);
        fout << token << OUTPUT_DELIMITER;
    }
    fout.close();
}

void dump_tokenization_to_file(string tokenized_file, vector<int> tokens) {
    ofstream fout(tokenized_file, ios::binary);
    for (int i : tokens) fout << i << " ";
    fout << endl;
    fout.close();
}

int main() {

    int vocab_size = 40000;

    // Load the corpus
    string input_corpus;
    stringstream buf;

    for (int i = 0; i < NUM_FILES; i++) {
        ifstream fin(CORPUS_DIR + to_string(i) + ".txt", ios::binary);
        buf << fin.rdbuf();
        fin.close();
    }

    input_corpus = buf.str();
    
    // print corpus length
    cout << "Corpus length: " << input_corpus.size() << endl;

    // Training
    tie(initial_vocab, reverse_initial_vocab) = generate_initial_vocab();
    vocab = initial_vocab;
    // map<vector<int>, int> frequencies = pre_tokenize(input_corpus, reverse_initial_vocab);
    vector<int> ints_corpus = to_ints(input_corpus, reverse_initial_vocab);
    train(ints_corpus, vocab_size);

    dump_vocab_to_file(VOCAB_OUTPUT);
    cout << "Saved tokens to file: " << VOCAB_OUTPUT << endl;

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
