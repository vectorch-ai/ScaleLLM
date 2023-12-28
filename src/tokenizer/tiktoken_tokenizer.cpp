#include "tiktoken_tokenizer.h"

#include <absl/strings/escaping.h>
#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>

#include <fstream>
#include <optional>

#include "common/logging.h"

namespace llm {

TiktokenTokenizer::TiktokenTokenizer(const std::string& vocab_file_path)
    : vocab_file_path_(vocab_file_path) {
  // read token + rank from vocab file
  std::ifstream fs(vocab_file_path);
  if (!fs) {
    GLOG(FATAL) << "Failed to open vocab file: " << vocab_file_path;
  }

  std::string line;
  while (std::getline(fs, line)) {
    if (line.empty()) {
      // skip empty line
      continue;
    }
    // split line by space
    const std::vector<std::string> parts = absl::StrSplit(line, ' ');
    if (parts.size() != 2) {
      GLOG(WARNING) << "Failed to parse line: " << line;
      continue;
    }
    // parse token and rank
    std::string token;
    if (!absl::Base64Unescape(parts[0], &token)) {
      GLOG(WARNING) << "Failed to parse token: " << parts[0];
      continue;
    }
    int32_t rank = 0;
    if (!absl::SimpleAtoi(parts[1], &rank)) {
      GLOG(WARNING) << "Failed to parse rank: " << parts[1];
      continue;
    }
    // TODO: check duplications
    encoder_[token] = rank;
    decoder_[rank] = token;
  }
}

void TiktokenTokenizer::byte_pair_encode(const std::string_view& piece,
                                         std::vector<int32_t>* ids) const {
  // This is a vector of (start, rank) pairs.
  // The rank is of the byte pair startig at position start.
  // The rank of the last item in the vector is not a valid value.
  std::vector<std::pair<int32_t, int32_t>> parts;
  parts.reserve(piece.size() + 1);
  const int32_t kMaxRank = std::numeric_limits<int32_t>::max();
  for (int32_t i = 0; i <= piece.size(); ++i) {
    parts.emplace_back(i, kMaxRank);
  }

  auto get_rank = [&piece, &parts, this](
                      int32_t start, int32_t skip) -> std::optional<int32_t> {
    if (start + skip + 2 < parts.size()) {
      auto s = parts[start].first;
      auto e = parts[start + skip + 2].first;
      const std::string key(piece.substr(s, e - s));
      auto it = encoder_.find(key);
      if (it != encoder_.end()) {
        return it->second;
      }
    }
    return std::nullopt;
  };

  // We look up the ranks once in the beginning and iteratively update
  // them during each merge, which reduces the number of rank lookups.
  for (int32_t i = 0; i < parts.size() - 2; ++i) {
    const auto rank = get_rank(i, 0);
    if (rank.has_value()) {
      // kMaxRank is a sentinel value and cannot be a valid rank.
      CHECK(rank.value() != kMaxRank) << "Invalid rank";
      parts[i].second = rank.value();
    }
  }

  while (parts.size() > 1) {
    // find i with min rank. kMaxRank is a sentinel value.
    int32_t min_rank = kMaxRank;
    int32_t min_i = 0;
    for (int32_t i = 0; i < parts.size() - 1; ++i) {
      const auto rank = parts[i].second;
      if (rank < min_rank) {
        min_rank = rank;
        min_i = i;
      }
    }

    if (min_rank == kMaxRank) {
      // No more merges possible.
      break;
    }

    // remove parts[min_i + 1].
    parts[min_i].second = get_rank(min_i, 1).value_or(kMaxRank);
    if (min_i > 0) {
      parts[min_i - 1].second = get_rank(min_i - 1, 1).value_or(kMaxRank);
    }
    parts.erase(parts.begin() + min_i + 1);
  }

  for (int32_t i = 0; i < parts.size() - 1; ++i) {
    const auto s = parts[i].first;
    const auto e = parts[i + 1].first;
    // get rank for each piece
    const std::string key(piece.substr(s, e - s));
    auto it = encoder_.find(key);
    if (it == encoder_.end()) {
      GLOG(ERROR) << "Failed to find key: " << key;
    } else {
      ids->push_back(it->second);
    }
  }
}

bool TiktokenTokenizer::encode(const std::string_view& text,
                               std::vector<int32_t>* ids) const {
  byte_pair_encode(text, ids);
  return true;
}

std::string TiktokenTokenizer::decode(const std::vector<int32_t>& ids) const {
  std::stringstream ss;
  for (const auto& id : ids) {
    auto it = decoder_.find(id);
    if (it == decoder_.end()) {
      GLOG(ERROR) << "Failed to find id: " << id;
    } else {
      ss << it->second;
    }
  }
  return ss.str();
}

size_t TiktokenTokenizer::vocab_size() const { return encoder_.size(); }

std::unique_ptr<Tokenizer> TiktokenTokenizer::clone() const {
  return std::make_unique<TiktokenTokenizer>(vocab_file_path_);
}

}  // namespace llm
