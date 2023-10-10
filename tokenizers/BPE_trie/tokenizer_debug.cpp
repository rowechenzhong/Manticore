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
#define LOW_MERGE_CUTOFF 5 // minimum frequency to merge
#define ll long long

pair<vector<string>, map<string,int>> generate_initial_vocab(){
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
    // initial_vocab_lengths = vector<int>(initial_vocab.size(), 1);
    // return make_tuple(initial_vocab, reverse_initial_vocab, initial_vocab_lengths);
    return make_pair(initial_vocab, reverse_initial_vocab);
}


// pair<vector<vector<byte>>, vector<int>>
map<vector<int>, int> pre_tokenize(string& corpus, map<string, int> reverse_initial_vocab) {
    /**
     * This method should preform pre-tokenization.
     * First, the corpus should be split by whitespace. Each distinct word should then be mapped to its frequency.
     * Then, each letter in each word should be mapped to its byte (the base vocabulary is 256).
    */
    map<vector<int>, int> frequencies; // maps each distinct word to its frequency.
    for (int i = 0; i < corpus.size(); ++i) {
        if (i % 1000 == 0)
            cout << "Pre-tokenizing: " << i << " of " << corpus.size() << " frequency size is " << frequencies.size() << "                   \r";
        int j = i;
        while (j < corpus.size() && corpus[j] != ' ' && corpus[j] != '\n' && corpus[j] != '\t') {
            j += 1;
        }
        string word = corpus.substr(i, j - i);
        vector<int> bytes;
        for (char c : word) {
            if (reverse_initial_vocab.find(string(1, c)) == reverse_initial_vocab.end()) {
                bytes.push_back(reverse_initial_vocab["<unk>"]);
            } else {
                bytes.push_back(reverse_initial_vocab[string(1, c)]);
            }
        }

        frequencies[bytes] += 1;
        i = j;
    }
    cout << endl;
    return frequencies;
}

void train(map<vector<int>, int>& corpus, unsigned int vocab_size) {
    vector<vector<int>> frequencies(vocab_size, vector<int>(vocab_size, 0));
    
    
    // map<int, vector<pair<int, int>>> pq; // maps frequency to all pairs of indices that have that frequency.

    // i don't care about removing... so I'll just use a priorityqueue.

    priority_queue<pair<int, pair<pair<int, int>, int>>> pq; // contains: (frequency, ((i,j), last update time))
    
    // vector of last update time
    vector<int> last_update_time(vocab_size, vocab.size());

    for (auto& [word, freq] : corpus) {
        for (int i = 0; i + 1 < word.size(); ++i) {
            // cout << "word has length " << word.size() << endl << endl;
            // cout << "i is " << i << " and word[i] is " << word[i] << endl;
            // cout << "word[i] is " << word[i] << " and word[i+1] is " << word[i+1] << endl;
            // // print frequencies size 
            // cout << "frequencies size is " << frequencies.size() << " by " << frequencies[0].size() << endl;
            frequencies[word[i]][word[i + 1]] += freq;
        }
    }

    for(int i = 0; i < vocab_size; ++i){
        for(int j = 0; j < vocab_size; ++j){
            if(frequencies[i][j] >= LOW_MERGE_CUTOFF && vocab[i].size() + vocab[j].size() <= MAX_TOKEN){
                pq.push(make_pair(frequencies[i][j], make_pair(make_pair(i, j), vocab.size())));
            }
        }
    }

    // int max_i = -1;
    // int max_j = -1;
    // int max_freq = 0;

    // for (int i = 0; i < vocab.size(); ++i) {
    //     for (int j = 0; j < vocab.size(); ++j) {
    //         if (frequencies[i][j] > max_freq && vocab[i].size() + vocab[j].size() <= MAX_TOKEN) {
    //             max_freq = frequencies[i][j];
    //             max_i = i;
    //             max_j = j;
    //         }
    //     }
    // }

    while (vocab.size() < vocab_size) {
        ll corpus_size = 0;
        for (auto& [word, freq] : corpus) {
            corpus_size += (ll)freq * (ll)word.size();
        }
        cout << "Current vocab size=" << vocab.size() << " current corpus size=" << corpus_size << "                        \r";

        // if(pq.empty()){
        //     cout << "No more merges possible. Stopping." << endl;
        //     break;
        // }
        // int max_i = pq.top().second.first;
        // int max_j = pq.top().second.second;


        // if (max_freq == 0) {
        //     cout << "No more merges possible. Stopping." << endl;
        //     break;
        // }

        // if(pq.empty()){
        //     cout << endl << "No more merges possible. Stopping.asd" << endl;
        //     break;
        // }
        
        // vocab.push_back(vocab[max_i] + vocab[max_j]);
        int max_i = -1;
        int max_j = -1;
        while(!pq.empty()){
            pair<int, pair<pair<int, int>, int>> top = pq.top();
            pq.pop();
            int i = top.second.first.first;
            int j = top.second.first.second;
            int last_update = top.second.second;
            if(last_update_time[i] <= last_update || last_update_time[j] <= last_update){ // Good!!!
                max_i = i;
                max_j = j;
                break;
            }
        }

        if(pq.empty()){
            cout << endl << "No more merges possible. Stopping." << endl;
            break;
        }
        // Updates last_update_time
        last_update_time[max_i] = vocab.size() + 1;
        last_update_time[max_j] = vocab.size() + 1;

        // Creates new word
        vocab.push_back(vocab[max_i] + vocab[max_j]);
        int new_vocab_index = vocab.size() - 1;

        // updates the corpus with the new word
        map<vector<int>, int> new_corpus;
        for (auto& [word, freq] : corpus) {
            vector<int> new_word;
            for (int i = 0; i < word.size(); ++i) {
                if (i < (int)word.size() - 1 && word[i] == max_i && word[i + 1] == max_j) {
                    new_word.push_back((int)vocab.size() - 1);
                    i += 1;
                } else {
                    new_word.push_back(word[i]);
                }
            }
            new_corpus[new_word] += freq;
        }
        corpus = new_corpus;

        // Re-calculates frequencies
        frequencies[max_i][max_j] = 0;
        frequencies[max_j] = vector<int>(vocab_size, 0);
        for(int i = 0; i < vocab_size; ++i) {
            frequencies[i][max_i] = 0;
        }
        for (auto& [word, freq] : corpus) {
            for (int i = 0; i + 1 < word.size(); ++i) {
                if (word[i] == max_j || word[i + 1] == max_i || word[i] == new_vocab_index || word[i+1] == new_vocab_index) { // We need to re-calculate these guys.
                    frequencies[word[i]][word[i + 1]] += freq;
                }
            }
        }

        // Re-calculates pq

        // int new_max_i = -1;
        // int new_max_j = -1;
        // max_freq = 0;
        for(int i = 0; i < vocab_size; ++i) {
            if(frequencies[i][max_i] > LOW_MERGE_CUTOFF && vocab[i].size() + vocab[max_i].size() <= MAX_TOKEN){
                // max_freq = frequencies[i][max_i];
                // new_max_i = i;
                // new_max_j = max_i;
                pq.push(make_pair(frequencies[i][max_i], make_pair(make_pair(i, max_i), vocab.size())));
            }
            if(frequencies[max_j][i] > LOW_MERGE_CUTOFF && vocab[i].size() + vocab[max_j].size() <= MAX_TOKEN){
                // max_freq = frequencies[max_j][i];
                // new_max_i = max_j;
                // new_max_j = i;
                pq.push(make_pair(frequencies[max_j][i], make_pair(make_pair(max_j, i), vocab.size())));
            }
            if(frequencies[i][new_vocab_index] > LOW_MERGE_CUTOFF && vocab[i].size() + vocab[new_vocab_index].size() <= MAX_TOKEN){
                // max_freq = frequencies[i][new_vocab_index];
                // new_max_i = i;
                // new_max_j = new_vocab_index;
                pq.push(make_pair(frequencies[i][new_vocab_index], make_pair(make_pair(i, new_vocab_index), vocab.size())));
            }
            if(frequencies[new_vocab_index][i] > LOW_MERGE_CUTOFF && vocab[i].size() + vocab[new_vocab_index].size() <= MAX_TOKEN){
                // max_freq = frequencies[new_vocab_index][i];
                // new_max_i = new_vocab_index;
                // new_max_j = i;
                pq.push(make_pair(frequencies[new_vocab_index][i], make_pair(make_pair(new_vocab_index, i), vocab.size())));
            }
        }
        // max_i = new_max_i;
        // max_j = new_max_j;
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

    // Example usage: loading the corpus
    // ifstream fin1("..\\..\\corpus\\communistmanifesto.txt");
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


    
    // print corpus length
    cout << "Corpus length: " << input_corpus.size() << endl;

    // // Debugging: print input corpus
    // cout << "Input corpus: " << input_corpus << endl;

    // Training
    tie(initial_vocab, reverse_initial_vocab) = generate_initial_vocab();
    vocab = initial_vocab;
    map<vector<int>, int> frequencies = pre_tokenize(input_corpus, reverse_initial_vocab);

    // // Debugging: print frequencies
    // cout << "Frequencies: " << endl;
    // for (auto& [token, freq] : frequencies) {
    //     for (int t : token) {
    //         cout << t << " ";
    //     }
    //     cout << ": " << freq << endl;
    // }

    // print frequencies size 
    // cout << "Frequencies size: " << frequencies.size() << endl;
    train(frequencies, vocab_size);

    // Tests
    vector<int> ints_corpus = tokenize(input_corpus);
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

    cout << endl;

    // dump_vocab_to_file("communistmanifesto_size4000_cap10.txt");
    if(WHICH_CORPUS == "communistmanifesto")
        dump_vocab_to_file("communistmanifesto_size" + to_string(vocab_size) + "_cap" + to_string(MAX_TOKEN) + ".txt");
    else if(WHICH_CORPUS == "mahabharata")
        dump_vocab_to_file("mahabharata_size" + to_string(vocab_size) + "_cap" + to_string(MAX_TOKEN) + ".txt");
    return 0;
}
