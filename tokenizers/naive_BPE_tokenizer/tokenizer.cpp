#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <cassert>

using namespace std;

vector<string> initial_vocab;
vector<string> vocab;

#define MAX_TOKEN 10

vector<int> to_ints(string& corpus, unordered_map<string, int>& reverse_initial_vocab) {
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
        cout << "Current vocab size=" << vocab.size() << " current corpus size=" << corpus.size() << "                        \r";
        vector<vector<int>> frequencies(vocab.size(), vector<int>(vocab.size(), 0));
        for (size_t i = 0; i < corpus.size() - 1; ++i) {
            frequencies[corpus[i]][corpus[i + 1]] += 1;
        }

        int max_frequency = 0;
        int max_i = 0;
        int max_j = 0;

        for (size_t i = 0; i < vocab.size(); ++i) {
            for (size_t j = 0; j < vocab.size(); ++j) {
                if (frequencies[i][j] > max_frequency && vocab[i].size() + vocab[j].size() <= MAX_TOKEN) {
                    max_frequency = frequencies[i][j];
                    max_i = i;
                    max_j = j;
                }
            }
        }

        vocab.push_back(vocab[max_i] + vocab[max_j]);

        vector<int> new_corpus;
        size_t ptr = 0;

        while (ptr < corpus.size()) {
            if (ptr < corpus.size() - 1 && corpus[ptr] == max_i && corpus[ptr + 1] == max_j) {
                new_corpus.push_back(vocab.size() - 1);
                ptr += 2;
            } else {
                new_corpus.push_back(corpus[ptr]);
                ptr += 1;
            }
        }
        corpus = new_corpus;
    }
    cout << endl;
}

vector<int> tokenize(string& corpus) {
    vector<int> tokenized_corpus;
    size_t ptr = 0;

    sort(vocab.begin(), vocab.end());

    while (ptr < corpus.size()) {
        if ((ptr & 0xfff) == 0) 
            cout << "Tokenizing: " << ptr << " of " << corpus.size() << "                   \r";
        int token = 0;
        size_t longest = 0;
        for (size_t i = 0; i < vocab.size(); ++i) {
            if (corpus.substr(ptr, MAX_TOKEN).compare(0, vocab[i].length(), vocab[i]) == 0) {
                if (vocab[i].length() > longest) {
                    longest = vocab[i].length();
                    token = i;
                }
            }
        }

        if (longest == 0) {
            ptr += 1;
        } else {
            tokenized_corpus.push_back(token);
            ptr += longest;
        }
    }
    cout << endl;

    return tokenized_corpus;
}

string detokenize(vector<int>& tokenized_corpus) {
    string corpus = "";
    for (int token : tokenized_corpus) {
        corpus += vocab[token];
    }
    return corpus;
}

void dump_vocab_to_file(string vocab_file) {
    ofstream fout(vocab_file);
    for (string token : vocab) {
        assert(token.size() <= MAX_TOKEN);
        fout << "<BRUH>" << token;
    }
    fout.close();
}

int main() {
    int vocab_size = 4000;

    // Initialize initial_vocab and reverse_initial_vocab.
    initial_vocab.push_back("<unk>");
    
    unordered_map<string, int> reverse_initial_vocab;
    for (int i = 0; i < 256; ++i) {
        string s(1, (char)i);
        initial_vocab.push_back(s);
        reverse_initial_vocab[s] = i + 1;
    }

    // Initialize vocab with initial_vocab.
    vocab = initial_vocab;

    // Example usage:
    // read input_corpus from atoms.txt
    string input_corpus;

    string WHICH_CORPUS = "mahabharata";
    stringstream buf;

    
    if(WHICH_CORPUS == "communistmanifesto"){
        ifstream fin1("C:\\Users\\rowec\\Documents\\GitHub\\Manticore\\corpus\\communistmanifesto.txt");
        buf << fin1.rdbuf();
        fin1.close();
    }else if(WHICH_CORPUS == "mahabharata"){
        ifstream fin1("C:\\Users\\rowec\\Documents\\GitHub\\Manticore\\corpus\\mahabharata1.txt");
        ifstream fin2("C:\\Users\\rowec\\Documents\\GitHub\\Manticore\\corpus\\mahabharata2.txt");
        ifstream fin3("C:\\Users\\rowec\\Documents\\GitHub\\Manticore\\corpus\\mahabharata3.txt");
        buf << fin1.rdbuf() << fin2.rdbuf() << fin3.rdbuf();
        fin1.close();
        fin2.close();
        fin3.close();
    }
    input_corpus = buf.str();

    vector<int> ints_corpus = to_ints(input_corpus, reverse_initial_vocab);
    train(ints_corpus, vocab_size);
    // vector<int> tokenized = tokenize(input_corpus);
    // string detokenized = detokenize(tokenized);

    //cout << "Input Corpus: " << input_corpus << endl;
    //cout << "Tokenized: ";
    // print the first 100 tokens
    // for (int token : tokenized) {
    //     cout << token << " ";
    // }
    
    // compression ratio
    cout << "Compression ratio:" << (double)ints_corpus.size() / (double)input_corpus.size() << endl;

    // print the distribution of token lengths.
    vector<int> token_lengths(MAX_TOKEN, 0);
    for (string token : vocab) {
        token_lengths[token.size()] += 1;
    }
    for (int i = 0; i < MAX_TOKEN; ++i) {
        cout << "Token length " << i << ": " << token_lengths[i] << endl;
    }

    cout << endl;

    // dump_vocab_to_file("communistmanifesto_size4000_cap10.txt");
    dump_vocab_to_file("mahabharata1_size4000_cap10.txt");
    return 0;
}
